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

#include "reverb/cc/reverb_service_impl.h"

#include <algorithm>
#include <list>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "reverb/cc/checkpointing/interface.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/support/cleanup.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/queue.h"
#include "reverb/cc/support/uint128.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {
namespace {

inline grpc::Status TableNotFound(absl::string_view name) {
  return grpc::Status(grpc::StatusCode::NOT_FOUND,
                      absl::StrCat("Priority table ", name, " was not found"));
}

inline grpc::Status Internal(const std::string& message) {
  return grpc::Status(grpc::StatusCode::INTERNAL, message);
}

}  // namespace

ReverbServiceImpl::ReverbServiceImpl(
    std::shared_ptr<CheckpointerInterface> checkpointer)
    : checkpointer_(std::move(checkpointer)) {}

tensorflow::Status ReverbServiceImpl::Create(
    std::vector<std::shared_ptr<Table>> tables,
    std::shared_ptr<CheckpointerInterface> checkpointer,
    std::unique_ptr<ReverbServiceImpl>* service) {
  // Can't use make_unique because it can't see the Impl's private constructor.
  auto new_service = std::unique_ptr<ReverbServiceImpl>(
      new ReverbServiceImpl(std::move(checkpointer)));
  TF_RETURN_IF_ERROR(new_service->Initialize(std::move(tables)));
  std::swap(new_service, *service);
  return tensorflow::Status::OK();
}

tensorflow::Status ReverbServiceImpl::Create(
    std::vector<std::shared_ptr<Table>> tables,
    std::unique_ptr<ReverbServiceImpl>* service) {
  return Create(std::move(tables), /*checkpointer=*/nullptr, service);
}

tensorflow::Status ReverbServiceImpl::Initialize(
    std::vector<std::shared_ptr<Table>> tables) {
  if (checkpointer_ != nullptr) {
    auto status = checkpointer_->LoadLatest(&chunk_store_, &tables);
    if (!status.ok() && !tensorflow::errors::IsNotFound(status)) {
      return status;
    }
  }

  for (auto& table : tables) {
    tables_[table->name()] = std::move(table);
  }

  tables_state_id_ = absl::MakeUint128(absl::Uniform<uint64_t>(rnd_),
                                       absl::Uniform<uint64_t>(rnd_));

  return tensorflow::Status::OK();
}

grpc::Status ReverbServiceImpl::Checkpoint(grpc::ServerContext* context,
                                           const CheckpointRequest* request,
                                           CheckpointResponse* response) {
  if (checkpointer_ == nullptr) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "no Checkpointer configured for the replay service.");
  }

  std::vector<Table*> tables;
  for (auto& table : tables_) {
    tables.push_back(table.second.get());
  }

  auto status = checkpointer_->Save(std::move(tables), 1,
                                    response->mutable_checkpoint_path());
  if (!status.ok()) return ToGrpcStatus(status);

  REVERB_LOG(REVERB_INFO) << "Stored checkpoint to "
                          << response->checkpoint_path();
  return grpc::Status::OK;
}

grpc::Status ReverbServiceImpl::InsertStream(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<InsertStreamResponse, InsertStreamRequest>*
        stream) {
  return InsertStreamInternal(context, stream);
}

grpc::Status ReverbServiceImpl::InsertStreamInternal(
    grpc::ServerContext* context,
    grpc::ServerReaderWriterInterface<InsertStreamResponse,
                                      InsertStreamRequest>* stream) {
  // Start a background thread that unpacks the data ahead of time.
  deepmind::reverb::internal::Queue<InsertStreamRequest> queue(1);
  auto read_thread = internal::StartThread("ReadThread", [stream, &queue]() {
    InsertStreamRequest request;
    while (stream->Read(&request) && queue.Push(std::move(request))) {
      request = InsertStreamRequest();
    }
    queue.SetLastItemPushed();
  });
  auto cleanup = internal::MakeCleanup([&queue] { queue.Close(); });

  absl::flat_hash_map<ChunkStore::Key, std::shared_ptr<ChunkStore::Chunk>>
      chunks;

  InsertStreamRequest request;
  while (queue.Pop(&request)) {
    if (request.has_chunk()) {
      ChunkStore::Key key = request.chunk().chunk_key();
      std::shared_ptr<ChunkStore::Chunk> chunk =
          chunk_store_.Insert(std::move(*request.mutable_chunk()));
      if (!chunk) {
        return grpc::Status(grpc::StatusCode::CANCELLED,
                            "Service has been closed");
      }
      chunks[key] = std::move(chunk);
    } else if (request.has_item()) {
      Table::Item item;

      auto push_or = [&chunks, &item](ChunkStore::Key key) -> grpc::Status {
        auto it = chunks.find(key);
        if (it == chunks.end()) {
          return Internal(
              absl::StrCat("Could not find sequence chunk ", key, "."));
        }
        item.chunks.push_back(it->second);
        return grpc::Status::OK;
      };

      for (ChunkStore::Key key : request.item().item().chunk_keys()) {
        auto status = push_or(key);
        if (!status.ok()) return status;
      }

      const auto& table_name = request.item().item().table();
      Table* table = PriorityTableByName(table_name);
      if (table == nullptr) return TableNotFound(table_name);

      const auto item_key = request.item().item().key();
      item.item = std::move(*request.mutable_item()->mutable_item());

      if (auto status = table->InsertOrAssign(item); !status.ok()) {
        return ToGrpcStatus(status);
      }

      // Let caller know that the item has been inserted if requested by the
      // caller.
      if (request.item().send_confirmation()) {
        InsertStreamResponse response;
        response.set_key(item_key);
        if (!stream->Write(response)) {
          return Internal(absl::StrCat(
              "Error when sending confirmation that item ", item_key,
              " has been successfully inserted/updated."));
        }
      }

      // Only keep specified chunks.
      absl::flat_hash_set<int64_t> keep_keys{
          request.item().keep_chunk_keys().begin(),
          request.item().keep_chunk_keys().end()};
      for (auto it = chunks.cbegin(); it != chunks.cend();) {
        if (keep_keys.find(it->first) == keep_keys.end()) {
          chunks.erase(it++);
        } else {
          ++it;
        }
      }
      REVERB_CHECK_EQ(chunks.size(), keep_keys.size())
          << "Kept less chunks than expected.";
    }
  }

  return grpc::Status::OK;
}

grpc::Status ReverbServiceImpl::MutatePriorities(
    grpc::ServerContext* context, const MutatePrioritiesRequest* request,
    MutatePrioritiesResponse* response) {
  Table* table = PriorityTableByName(request->table());
  if (table == nullptr) return TableNotFound(request->table());

  auto status = table->MutateItems(
      std::vector<KeyWithPriority>(request->updates().begin(),
                                   request->updates().end()),
      request->delete_keys());
  if (!status.ok()) return ToGrpcStatus(status);
  return grpc::Status::OK;
}

grpc::Status ReverbServiceImpl::Reset(grpc::ServerContext* context,
                                      const ResetRequest* request,
                                      ResetResponse* response) {
  Table* table = PriorityTableByName(request->table());
  if (table == nullptr) return TableNotFound(request->table());

  auto status = table->Reset();
  if (!status.ok()) {
    return ToGrpcStatus(status);
  }
  return grpc::Status::OK;
}

grpc::Status ReverbServiceImpl::SampleStream(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<SampleStreamResponse, SampleStreamRequest>*
        stream) {
  return SampleStreamInternal(context, stream);
}

grpc::Status ReverbServiceImpl::SampleStreamInternal(
    grpc::ServerContext* context,
    grpc::ServerReaderWriterInterface<SampleStreamResponse,
                                      SampleStreamRequest>* stream) {
  SampleStreamRequest request;
  if (!stream->Read(&request)) {
    return Internal("Could not read initial request");
  }
  int64_t timeout_ms = request.has_rate_limiter_timeout()
                         ? request.rate_limiter_timeout().milliseconds()
                         : -1;
  absl::Duration timeout = (timeout_ms < 0) ? absl::InfiniteDuration()
                                            : absl::Milliseconds(timeout_ms);

  do {
    if (request.num_samples() <= 0) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "`num_samples` must be > 0.");
    }
    if (request.flexible_batch_size() <= 0 &&
        request.flexible_batch_size() != Sampler::kAutoSelectValue) {
      return grpc::Status(
          grpc::StatusCode::INVALID_ARGUMENT,
          absl::StrCat("`flexible_batch_size` must be > 0 or ",
                       Sampler::kAutoSelectValue, " (for auto tuning)."));
    }
    Table* table = PriorityTableByName(request.table());
    if (table == nullptr) return TableNotFound(request.table());
    int32_t default_flexible_batch_size = table->DefaultFlexibleBatchSize();

    int count = 0;

    while (!context->IsCancelled() && count != request.num_samples()) {
      std::vector<Table::SampledItem> samples;
      int32_t max_batch_size = std::min<int32_t>(
          request.flexible_batch_size() == Sampler::kAutoSelectValue
              ? default_flexible_batch_size
              : request.flexible_batch_size(),
          request.num_samples() - count);
      if (auto status =
              table->SampleFlexibleBatch(&samples, max_batch_size, timeout);
          !status.ok()) {
        return ToGrpcStatus(status);
      }
      count += samples.size();

      for (auto& sample : samples) {
        for (int i = 0; i < sample.chunks.size(); i++) {
          SampleStreamResponse response;
          response.set_end_of_sequence(i + 1 == sample.chunks.size());

          // Attach the info to the first message.
          if (i == 0) {
            *response.mutable_info()->mutable_item() = sample.item;
            response.mutable_info()->set_probability(sample.probability);
            response.mutable_info()->set_table_size(sample.table_size);
          }

          // We const cast to avoid copying the proto.
          response.set_allocated_data(
              const_cast<ChunkData*>(&sample.chunks[i]->data()));

          grpc::WriteOptions options;
          options.set_no_compression();  // Data is already compressed.
          bool ok = stream->Write(response, options);
          response.release_data();
          if (!ok) {
            return Internal("Failed to write to Sample stream.");
          }

          // We no longer need our chunk reference, so we free it.
          sample.chunks[i] = nullptr;
        }
      }
    }

    request.Clear();
  } while (stream->Read(&request));

  return grpc::Status::OK;
}

Table* ReverbServiceImpl::PriorityTableByName(absl::string_view name) const {
  auto it = tables_.find(name);
  if (it == tables_.end()) return nullptr;
  return it->second.get();
}

void ReverbServiceImpl::Close() {
  for (auto& table : tables_) {
    table.second->Close();
  }
}

grpc::Status ReverbServiceImpl::ServerInfo(grpc::ServerContext* context,
                                           const ServerInfoRequest* request,
                                           ServerInfoResponse* response) {
  for (const auto& iter : tables_) {
    *response->add_table_info() = iter.second->info();
  }
  *response->mutable_tables_state_id() = Uint128ToMessage(tables_state_id_);
  return grpc::Status::OK;
}

absl::flat_hash_map<std::string, std::shared_ptr<Table>>
ReverbServiceImpl::tables() const {
  return tables_;
}

}  // namespace reverb
}  // namespace deepmind
