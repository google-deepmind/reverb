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

#include "reverb/cc/replay_client.h"

#include <algorithm>
#include <memory>

#include "grpcpp/support/channel_arguments.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "reverb/cc/platform/grpc_utils.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/replay_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/uint128.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace deepmind {
namespace reverb {
namespace {

constexpr int kMaxMessageSize = 30 * 1000 * 1000;

grpc::ChannelArguments CreateChannelArguments() {
  grpc::ChannelArguments arguments;
  arguments.SetMaxReceiveMessageSize(kMaxMessageSize);
  arguments.SetMaxSendMessageSize(kMaxMessageSize);
  arguments.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 30 * 1000);
  arguments.SetLoadBalancingPolicyName("round_robin");
  return arguments;
}

}  // namespace

ReplayClient::ReplayClient(
    std::shared_ptr</* grpc_gen:: */ReplayService::StubInterface> stub)
    : stub_(std::move(stub)) {
  REVERB_CHECK(stub_ != nullptr);
}

ReplayClient::ReplayClient(absl::string_view server_address)
    : stub_(/* grpc_gen:: */ReplayService::NewStub(CreateCustomGrpcChannel(
        server_address, MakeChannelCredentials(), CreateChannelArguments()))) {}

tensorflow::Status ReplayClient::MaybeUpdateServerInfoCache(
    absl::Duration timeout,
    std::shared_ptr<internal::FlatSignatureMap>* cached_flat_signatures) {
  // TODO(b/154927570): Once tables can be mutated on the server, we'll need to
  // decide a new rule for updating the server info, instead of doing it just
  // once at the beginning.
  {
    // Exit early if we have table info cached.
    absl::ReaderMutexLock lock(&cached_table_mu_);
    if (cached_flat_signatures_) {
      *cached_flat_signatures = cached_flat_signatures_;
      return tensorflow::Status::OK();
    }
  }

  // This performs an RPC, so don't run it within a mutex.
  // Note, this operation can run into a race condition where multiple
  // threads of the same ReplayClient request server info, get different
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
  TF_RETURN_IF_ERROR(GetServerInfo(timeout, &info));

  absl::MutexLock lock(&cached_table_mu_);
  TF_RETURN_IF_ERROR(LockedUpdateServerInfoCache(info));
  *cached_flat_signatures = cached_flat_signatures_;
  return tensorflow::Status::OK();
}

tensorflow::Status ReplayClient::NewWriter(
    int chunk_length, int max_timesteps, bool delta_encoded,
    std::unique_ptr<ReplayWriter>* writer) {
  // TODO(b/154928265): caching this request?  For example, if
  // it's been N seconds or minutes, it may be time to
  // get an updated ServerInfo and see if there are new tables.
  std::shared_ptr<internal::FlatSignatureMap> cached_flat_signatures;
  // TODO(b/154927687): It is not ideal that this blocks forever. We should
  // probably limit this and ignore the signature if it couldn't be found within
  // some limits.
  TF_RETURN_IF_ERROR(MaybeUpdateServerInfoCache(absl::InfiniteDuration(),
                                                &cached_flat_signatures));
  *writer = absl::make_unique<ReplayWriter>(stub_, chunk_length, max_timesteps,
                                            delta_encoded,
                                            std::move(cached_flat_signatures));
  return tensorflow::Status::OK();
}

tensorflow::Status ReplayClient::MutatePriorities(
    absl::string_view table, const std::vector<KeyWithPriority>& updates,
    const std::vector<uint64_t>& deletes) {
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
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

tensorflow::Status ReplayClient::NewSampler(
    const std::string& table, const ReplaySampler::Options& options,
    std::unique_ptr<ReplaySampler>* sampler) {
  *sampler = absl::make_unique<ReplaySampler>(stub_, table, options);
  return tensorflow::Status::OK();
}

tensorflow::Status ReplayClient::NewSampler(
    const std::string& table, const ReplaySampler::Options& options,
    const tensorflow::DataTypeVector& validation_dtypes,
    const std::vector<tensorflow::PartialTensorShape>& validation_shapes,
    absl::Duration validation_timeout,
    std::unique_ptr<ReplaySampler>* sampler) {
  // TODO(b/154928265): caching this request?  For example, if
  // it's been N seconds or minutes, it may be time to
  // get an updated ServerInfo and see if there are new tables.
  std::shared_ptr<internal::FlatSignatureMap> cached_flat_signatures;
  TF_RETURN_IF_ERROR(
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
  } else {
    const auto& dtypes_and_shapes_no_info = iter->second;
    // Only perform check if the table had a signature associated with it.
    if (dtypes_and_shapes_no_info) {
      std::vector<tensorflow::DtypeAndPartialTensorShape> dtypes_and_shapes;
      // First element of sampled signature is the key.
      dtypes_and_shapes.push_back(
          {tensorflow::DT_UINT64, tensorflow::PartialTensorShape({})});
      // Second element of sampled signature is the probability value.
      dtypes_and_shapes.push_back(
          {tensorflow::DT_DOUBLE, tensorflow::PartialTensorShape({})});
      // Third element of sampled signature is the size of the table.
      dtypes_and_shapes.push_back(
          {tensorflow::DT_INT64, tensorflow::PartialTensorShape({})});
      for (const auto& dtype_and_shape : *dtypes_and_shapes_no_info) {
        dtypes_and_shapes.push_back(dtype_and_shape);
      }
      if (dtypes_and_shapes.size() != validation_shapes.size()) {
        return tensorflow::errors::InvalidArgument(
            "Inconsistent number of tensors requested from table '", table,
            "'.  Requested ", validation_shapes.size(),
            " tensors, but table signature shows ", dtypes_and_shapes.size(),
            " tensors.  Table signature: ",
            internal::DtypesShapesString(dtypes_and_shapes));
      }
      for (int i = 0; i < dtypes_and_shapes.size(); ++i) {
        if (dtypes_and_shapes[i].dtype != validation_dtypes[i] ||
            !dtypes_and_shapes[i].shape.IsCompatibleWith(
                validation_shapes[i])) {
          return tensorflow::errors::InvalidArgument(
              "Requested incompatible tensor at flattened index ", i,
              " from table '", table, "'.  Requested (dtype, shape): (",
              tensorflow::DataTypeString(validation_dtypes[i]), ", ",
              validation_shapes[i].DebugString(),
              ").  Signature (dtype, shape): (",
              tensorflow::DataTypeString(dtypes_and_shapes[i].dtype), ", ",
              dtypes_and_shapes[i].shape.DebugString(), ").  Table signature: ",
              internal::DtypesShapesString(dtypes_and_shapes));
        }
      }
    }
  }

  // TODO(b/154927849): Do sanity checks on the buffer_size and max_samples.
  // TODO(b/154928566): Maybe we don't even need to expose the buffer_size.
  return NewSampler(table, options, sampler);
}

tensorflow::Status ReplayClient::GetServerInfo(absl::Duration timeout,
                                               struct ServerInfo* info) {
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  if (timeout != absl::InfiniteDuration()) {
    context.set_deadline(std::chrono::system_clock::now() +
                         absl::ToChronoSeconds(timeout));
  }

  ServerInfoRequest request;
  ServerInfoResponse response;
  TF_RETURN_IF_ERROR(
      FromGrpcStatus(stub_->ServerInfo(&context, request, &response)));
  info->tables_state_id = MessageToUint128(response.tables_state_id());
  for (class TableInfo& table : *response.mutable_table_info()) {
    info->table_info.emplace_back(std::move(table));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ReplayClient::ServerInfo(struct ServerInfo* info) {
  return ServerInfo(absl::InfiniteDuration(), info);
}

tensorflow::Status ReplayClient::ServerInfo(absl::Duration timeout,
                                            struct ServerInfo* info) {
  struct ServerInfo local_info;
  TF_RETURN_IF_ERROR(GetServerInfo(timeout, &local_info));
  {
    absl::MutexLock lock(&cached_table_mu_);
    TF_RETURN_IF_ERROR(LockedUpdateServerInfoCache(local_info));
  }
  std::swap(*info, local_info);
  return tensorflow::Status::OK();
}

tensorflow::Status ReplayClient::LockedUpdateServerInfoCache(
    const struct ServerInfo& info) {
  if (!cached_flat_signatures_ || tables_state_id_ != info.tables_state_id) {
    internal::FlatSignatureMap signatures;
    for (const auto& table_info : info.table_info) {
      TF_RETURN_IF_ERROR(internal::FlatSignatureFromTableInfo(
          table_info, &(signatures[table_info.name()])));
    }
    cached_flat_signatures_.reset(
        new internal::FlatSignatureMap(std::move(signatures)));
    tables_state_id_ = info.tables_state_id;
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ReplayClient::Reset(const std::string& table) {
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  ResetRequest request;
  request.set_table(table);
  ResetResponse response;
  return FromGrpcStatus(stub_->Reset(&context, request, &response));
}

tensorflow::Status ReplayClient::Checkpoint(std::string* path) {
  grpc::ClientContext context;
  context.set_fail_fast(true);
  CheckpointRequest request;
  CheckpointResponse response;
  TF_RETURN_IF_ERROR(
      FromGrpcStatus(stub_->Checkpoint(&context, request, &response)));
  *path = response.checkpoint_path();
  return tensorflow::Status::OK();
}

}  // namespace reverb
}  // namespace deepmind
