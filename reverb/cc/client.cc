// Copyright 2019 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "reverb/cc/client.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "grpcpp/support/channel_arguments.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "third_party/grpc/include/grpcpp/security/credentials.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/patterns.pb.h"
#include "reverb/cc/platform/grpc_utils.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/streaming_trajectory_writer.h"
#include "reverb/cc/structured_writer.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/uint128.h"
#include "reverb/cc/trajectory_writer.h"

namespace deepmind {
namespace reverb {
namespace {

grpc::ChannelArguments CreateChannelArguments() {
  grpc::ChannelArguments arguments;
  arguments.SetMaxReceiveMessageSize(-1);  // Unlimited.
  arguments.SetMaxSendMessageSize(-1);     // Unlimited.
  arguments.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 30 * 1000);
  arguments.SetLoadBalancingPolicyName("round_robin");
  return arguments;
}

}  // namespace

Client::Client(std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub)
    : stub_(std::move(stub)) {
  REVERB_CHECK(stub_ != nullptr);
}

Client::Client(absl::string_view server_address)
    : stub_(/* grpc_gen:: */ReverbService::NewStub(grpc::CreateCustomChannel(
          std::string(server_address), MakeChannelCredentials(),
          CreateChannelArguments()))) {}

absl::Status Client::MaybeUpdateServerInfoCache(
    absl::Duration timeout,
    std::shared_ptr<internal::FlatSignatureMap>* cached_flat_signatures) {
  {
    // Exit early if we have table info cached.
    absl::ReaderMutexLock lock(cached_table_mu_);
    if (cached_flat_signatures_) {
      *cached_flat_signatures = cached_flat_signatures_;
      return absl::OkStatus();
    }
  }

  if (timeout == -absl::InfiniteDuration()) {
    // If timeout is -infinity, the user asked for data to be returned
    // immediately and without error; but we don't have anything already cached.
    // Just act like everything is fine! (Return empty signatures).
    *cached_flat_signatures = std::make_shared<internal::FlatSignatureMap>();
    return absl::OkStatus();
  }

  // This performs an RPC, so don't run it within a mutex.
  // Note, this operation can run into a race condition where multiple
  // threads of the same Client request server info, get different
  // values, and one of these overwrites cached_table_info_ with a staler
  // ServerInfo after another thread writes a newer version of ServerInfo
  // Then future writers see stale signatures.
  //
  // In practice this isn't a real issue because:
  //  (1) This type of concurrency is not common: once ServerInfo
  //      is set, this code path isn't executed again.
  //  (2) ServerInfo doesn't change often on a reverb server.
  //  (3) Due to the default gRPC client load balancing mechanism,
  //      a client with a stub connection to one IP of a group of
  //      servers will always use the same IP address for all
  //      consecutive requests.  So even concurrent requests will all
  //      go to the same server ("pick_first" policy):
  //
  //      https://github.com/grpc/grpc/blob/631fe79f84af295c60aea5693350b45154827398/src/core/ext/filters/client_channel/client_channel.cc#L1661
  struct ServerInfo info;
  REVERB_RETURN_IF_ERROR(GetServerInfo(timeout, &info));

  absl::MutexLock lock(cached_table_mu_);
  REVERB_RETURN_IF_ERROR(LockedUpdateServerInfoCache(info));
  *cached_flat_signatures = cached_flat_signatures_;
  return absl::OkStatus();
}

absl::Status Client::NewWriter(int chunk_length, int max_timesteps,
                               bool delta_encoded, int max_in_flight_items,
                               std::unique_ptr<Writer>* writer) {
  // TODO(b/154928265): caching this request?  For example, if
  // it's been N seconds or minutes, it may be time to
  // get an updated ServerInfo and see if there are new tables.
  std::shared_ptr<internal::FlatSignatureMap> cached_flat_signatures;
  // TODO(b/154927687): It is not ideal that this blocks forever. We should
  // probably limit this and ignore the signature if it couldn't be found within
  // some limits.
  REVERB_RETURN_IF_ERROR(MaybeUpdateServerInfoCache(absl::InfiniteDuration(),
                                                    &cached_flat_signatures));
  *writer = std::make_unique<Writer>(
      stub_, chunk_length, max_timesteps, delta_encoded,
      std::move(cached_flat_signatures), max_in_flight_items);
  return absl::OkStatus();
}

absl::Status Client::NewWriter(int chunk_length, int max_timesteps,
                               bool delta_encoded,
                               std::unique_ptr<Writer>* writer) {
  return NewWriter(chunk_length, max_timesteps, delta_encoded,
                   Writer::kDefaultMaxInFlightItems, writer);
}

absl::Status Client::MutatePriorities(
    absl::string_view table, const std::vector<KeyWithPriority>& updates,
    const std::vector<uint64_t>& deletes, absl::Duration timeout) {
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  context.set_deadline(absl::ToChronoTime(absl::Now() + timeout));
  MutatePrioritiesRequest request;
  request.set_table(table.data(), table.size());
  for (const KeyWithPriority& item : updates) {
    *request.add_updates() = item;
  }
  for (int64_t key : deletes) {
    request.add_delete_keys(key);
  }
  MutatePrioritiesResponse response;
  return FromGrpcStatus(stub_->MutatePriorities(&context, request, &response));
}

absl::Status Client::NewSampler(const std::string& table,
                                const Sampler::Options& options,
                                internal::DtypesAndShapes dtypes_and_shapes,
                                std::unique_ptr<Sampler>* sampler) {
  REVERB_RETURN_IF_ERROR(options.Validate());

  std::shared_ptr<Table> table_ptr;
  if (GetLocalTablePtr(table, &table_ptr).ok()) {
    REVERB_LOG_EVERY_POW_2(REVERB_INFO)
        << "Sampler and server are owned by the same process (" << getpid()
        << ") so Table " << table << " is accessed directly without gRPC.";
    *sampler = std::make_unique<Sampler>(std::move(table_ptr), options,
                                         std::move(dtypes_and_shapes));
  } else {
    *sampler = std::make_unique<Sampler>(stub_, table, options,
                                         std::move(dtypes_and_shapes));
  }

  return absl::OkStatus();
}

absl::Status Client::NewSampler(const std::string& table,
                                const Sampler::Options& options,
                                absl::Duration validation_timeout,
                                std::unique_ptr<Sampler>* sampler) {
  internal::DtypesAndShapes dtypes_and_shapes;
  auto status = GetDtypesAndShapesForSampler(table, validation_timeout,
                                             &dtypes_and_shapes);

  if (absl::IsDeadlineExceeded(status)) {
    REVERB_LOG(REVERB_WARNING)
        << "Unable to validate shapes and dtypes of new sampler for '" << table
        << "' as server could not be reached in time (" << validation_timeout
        << "). We were thus unable to fetch signature from server. The "
           "sampler will be constructed without validating the dtypes "
           "and shapes.";
  }

  return NewSampler(table, options, std::move(dtypes_and_shapes), sampler);
}

absl::Status Client::GetDtypesAndShapesForSampler(
    const std::string& table, absl::Duration validation_timeout,
    internal::DtypesAndShapes* dtypes_and_shapes) {
  // TODO(b/154928265): caching this request?  For example, if
  // it's been N seconds or minutes, it may be time to
  // get an updated ServerInfo and see if there are new tables.

  std::shared_ptr<internal::FlatSignatureMap> cached_flat_signatures;
  REVERB_RETURN_IF_ERROR(
      MaybeUpdateServerInfoCache(validation_timeout, &cached_flat_signatures));

  const auto iter = cached_flat_signatures->find(table);
  if (iter == cached_flat_signatures->end()) {
    std::vector<std::string> table_names;
    for (const auto& table : *cached_flat_signatures) {
      table_names.push_back(absl::StrCat("'", table.first, "'"));
    }
    REVERB_LOG(REVERB_WARNING)
        << "Unable to find table '" << table
        << "' in server signature.  Perhaps the table hasn't yet been added to "
           "the server?  Available tables: ["
        << absl::StrJoin(table_names, ", ") << "].";
    dtypes_and_shapes->reset();
  } else if (!iter->second) {
    // Found the table, but signature is empty.
    dtypes_and_shapes->reset();
  } else {
    // Found the table, and found a signature.
    const auto& old_dtypes_and_shapes = iter->second;
    std::vector<internal::TensorSpec> dtypes_and_shapes_vec;
    dtypes_and_shapes_vec.reserve(old_dtypes_and_shapes->size());
    for (const auto& dtype_and_shape : *old_dtypes_and_shapes) {
      dtypes_and_shapes_vec.push_back(dtype_and_shape);
    }

    dtypes_and_shapes->emplace(std::move(dtypes_and_shapes_vec));
  }
  return absl::OkStatus();
}

absl::Status Client::NewSampler(
    const std::string& table, const Sampler::Options& options,
    const tensorflow::DataTypeVector& validation_dtypes,
    const std::vector<tensorflow::PartialTensorShape>& validation_shapes,
    absl::Duration validation_timeout, std::unique_ptr<Sampler>* sampler) {
  if (validation_dtypes.size() != validation_shapes.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "validation_shapes.size() != validation_dtypes.size() (",
        validation_shapes.size(), " vs. ", validation_dtypes.size(), ")"));
  }

  internal::DtypesAndShapes dtypes_and_shapes;
  REVERB_RETURN_IF_ERROR(GetDtypesAndShapesForSampler(table, validation_timeout,
                                                      &dtypes_and_shapes));
  // Only perform check if the table had a signature associated with it.
  if (dtypes_and_shapes) {
    if (dtypes_and_shapes->size() != validation_shapes.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Inconsistent number of tensors requested from table '", table,
          "'.  Requested ", validation_shapes.size(),
          " tensors, but table signature shows ", dtypes_and_shapes->size(),
          " tensors.  Table signature: ",
          internal::DtypesShapesString(*dtypes_and_shapes)));
    }
    for (int i = 0; i < dtypes_and_shapes->size(); ++i) {
      if (dtypes_and_shapes->at(i).dtype != validation_dtypes[i] ||
          !dtypes_and_shapes->at(i).shape.IsCompatibleWith(
              validation_shapes[i])) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Requested incompatible tensor at flattened index ", i,
            " from table '", table, "'.  Requested (dtype, shape): (",
            tensorflow::DataTypeString(validation_dtypes[i]), ", ",
            validation_shapes[i].DebugString(),
            ").  Signature (dtype, shape): (",
            tensorflow::DataTypeString(dtypes_and_shapes->at(i).dtype), ", ",
            dtypes_and_shapes->at(i).shape.DebugString(),
            ").  Table signature: ",
            internal::DtypesShapesString(*dtypes_and_shapes)));
      }
    }
  } else {
    // dtypes_and_shapes lacks any signature info; build it from
    // the validation inputs.
    std::vector<internal::TensorSpec> dtypes_and_shapes_vec;
    dtypes_and_shapes_vec.reserve(validation_shapes.size());
    for (int i = 0; i < validation_shapes.size(); ++i) {
      dtypes_and_shapes_vec.push_back(
          {/*name=*/"?", validation_dtypes[i], validation_shapes[i]});
    }
    dtypes_and_shapes.emplace(std::move(dtypes_and_shapes_vec));
  }

  return NewSampler(table, options, std::move(dtypes_and_shapes), sampler);
}

absl::Status Client::NewSamplerWithoutSignatureCheck(
    const std::string& table, const Sampler::Options& options,
    std::unique_ptr<Sampler>* sampler) {
  return NewSampler(table, options, /*dtypes_and_shapes=*/absl::nullopt,
                    sampler);
}

absl::Status Client::GetServerInfo(absl::Duration timeout,
                                   struct ServerInfo* info) {
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  if (timeout != absl::InfiniteDuration()) {
    context.set_deadline(std::chrono::system_clock::now() +
                         absl::ToChronoSeconds(timeout));
  }

  ServerInfoRequest request;
  ServerInfoResponse response;
  REVERB_RETURN_IF_ERROR(
      FromGrpcStatus(stub_->ServerInfo(&context, request, &response)));
  info->tables_state_id = MessageToUint128(response.tables_state_id());
  for (class TableInfo& table : *response.mutable_table_info()) {
    info->table_info.emplace_back(std::move(table));
  }
  return absl::OkStatus();
}

absl::Status Client::ServerInfo(struct ServerInfo* info) {
  return ServerInfo(absl::InfiniteDuration(), info);
}

absl::Status Client::ServerInfo(absl::Duration timeout,
                                struct ServerInfo* info) {
  struct ServerInfo local_info;
  REVERB_RETURN_IF_ERROR(GetServerInfo(timeout, &local_info));
  {
    absl::MutexLock lock(cached_table_mu_);
    REVERB_RETURN_IF_ERROR(LockedUpdateServerInfoCache(local_info));
  }
  std::swap(*info, local_info);
  return absl::OkStatus();
}

absl::Status Client::LockedUpdateServerInfoCache(
    const struct ServerInfo& info) {
  if (!cached_flat_signatures_ || tables_state_id_ != info.tables_state_id) {
    internal::FlatSignatureMap signatures;
    for (const auto& table_info : info.table_info) {
      REVERB_RETURN_IF_ERROR(internal::FlatSignatureFromTableInfo(
          table_info, &(signatures[table_info.name()])));
    }
    cached_flat_signatures_.reset(
        new internal::FlatSignatureMap(std::move(signatures)));
    tables_state_id_ = info.tables_state_id;
  }
  return absl::OkStatus();
}

absl::Status Client::Reset(const std::string& table) {
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  ResetRequest request;
  request.set_table(table);
  ResetResponse response;
  return FromGrpcStatus(stub_->Reset(&context, request, &response));
}

absl::Status Client::Checkpoint(std::string* path) {
  grpc::ClientContext context;
  context.set_fail_fast(true);
  CheckpointRequest request;
  CheckpointResponse response;
  REVERB_RETURN_IF_ERROR(
      FromGrpcStatus(stub_->Checkpoint(&context, request, &response)));
  *path = response.checkpoint_path();
  return absl::OkStatus();
}

absl::Status Client::GetLocalTablePtr(absl::string_view table_name,
                                      std::shared_ptr<Table>* out) {
  grpc::ClientContext context;
  context.set_wait_for_ready(false);
  auto stream = stub_->InitializeConnection(&context);

  InitializeConnectionRequest request;
  request.set_pid(getpid());
  request.set_table_name(table_name.data(), table_name.size());
  if (!stream->Write(request)) {
    REVERB_RETURN_IF_ERROR(FromGrpcStatus(stream->Finish()));
    return absl::InternalError(
        "InitializeConnection: Failed to write to stream.");
  }

  InitializeConnectionResponse response;
  if (!stream->Read(&response)) {
    REVERB_RETURN_IF_ERROR(FromGrpcStatus(stream->Finish()));
    return absl::InternalError(
        "InitializeConnection: Failed to read from stream.");
  }

  if (response.address() == 0) {
    return absl::FailedPreconditionError(
        "Client and server are not running in the same process.");
  }

  *out = *reinterpret_cast<std::shared_ptr<Table>*>(response.address());
  request.set_ownership_transferred(true);
  stream->Write(request);

  return FromGrpcStatus(stream->Finish());
}

absl::Status Client::NewTrajectoryWriter(
    const TrajectoryWriter::Options& options,
    absl::Duration get_signature_timeout,
    std::unique_ptr<TrajectoryWriter>* writer) {
  std::shared_ptr<internal::FlatSignatureMap> cached_flat_signatures;
  REVERB_RETURN_IF_ERROR(MaybeUpdateServerInfoCache(get_signature_timeout,
                                                    &cached_flat_signatures));

  TrajectoryWriter::Options updated_options = options;
  updated_options.flat_signature_map = *cached_flat_signatures;
  return NewTrajectoryWriter(updated_options, writer);
}

absl::Status Client::NewTrajectoryWriter(
    const TrajectoryWriter::Options& options,
    std::unique_ptr<TrajectoryWriter>* writer) {
  REVERB_RETURN_IF_ERROR(options.Validate());
  *writer = std::make_unique<TrajectoryWriter>(stub_, options);
  return absl::OkStatus();
}

absl::Status Client::NewStreamingTrajectoryWriter(
    const TrajectoryWriter::Options& options,
    std::unique_ptr<StreamingTrajectoryWriter>* writer) {
  REVERB_RETURN_IF_ERROR(options.Validate());
  *writer = std::make_unique<StreamingTrajectoryWriter>(stub_, options);
  return absl::OkStatus();
}

absl::Status Client::NewStructuredWriter(
    std::vector<StructuredWriterConfig> configs,
    std::unique_ptr<StructuredWriter>* writer) {
  if (configs.empty()) {
    return absl::InvalidArgumentError("At least one config must be provided.");
  }

  // Keep track of the maximum history length required by any of the configs and
  // use this to build the `TrajectoryWriter` that the `StructuredWriter` will
  // wrap.
  int max_num_keep_alive_refs = 0;
  for (int i = 0; i < configs.size(); i++) {
    // Find the maximum history length required by this config.
    int num_keep_alive_refs = 0;
    for (const auto& node : configs[i].flat()) {
      num_keep_alive_refs = std::max(
          num_keep_alive_refs, std::abs(std::min(node.start(), node.stop())));
    }

    // Update the global maximum history length required.
    max_num_keep_alive_refs =
        std::max(max_num_keep_alive_refs, num_keep_alive_refs);

    // If we wish to avoid segfault then it is important that the buffers
    // contains enough steps for the pattern to be applied before anything is
    // attempted. We therefore check if the config already contains a condition
    // that ensures that the config is not applied prematurely. If none of the
    // existing conditions fulfill this responsibility then we create and add
    // one to the config.
    // Add a condition for the buffer length if it doesn't already have one.
    if (std::none_of(configs[i].conditions().begin(),
                     configs[i].conditions().end(), [&](const auto& c) {
                       return c.buffer_length() &&
                              c.ge() >= num_keep_alive_refs;
                     })) {
      auto* cond = configs[i].add_conditions();
      cond->set_buffer_length(true);
      cond->set_ge(num_keep_alive_refs);
    }

    if (auto status = ValidateStructuredWriterConfig(configs[i]);
        !status.ok()) {
      return absl::Status(
          status.code(),
          absl::StrFormat("Invalid configuration at position %d: %s", i,
                          status.message()));
    }
  }

  TrajectoryWriter::Options options = {
      .chunker_options =
          std::make_shared<AutoTunedChunkerOptions>(max_num_keep_alive_refs),
  };
  std::unique_ptr<TrajectoryWriter> trajectory_writer;
  REVERB_RETURN_IF_ERROR(NewTrajectoryWriter(options, &trajectory_writer));

  *writer = std::make_unique<StructuredWriter>(std::move(trajectory_writer),
                                               std::move(configs));

  return absl::OkStatus();
}

}  // namespace reverb
}  // namespace deepmind
