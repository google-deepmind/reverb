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

#include "reverb/cc/reverb_service_async_impl.h"

#include <algorithm>
#include <limits>
#include <list>
#include <memory>
#include <queue>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "reverb/cc/checkpointing/interface.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/reverb_server_bidi_reactor.h"
#include "reverb/cc/reverb_server_table_reactor.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/support/cleanup.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/trajectory_util.h"
#include "reverb/cc/support/uint128.h"
#include "reverb/cc/support/unbounded_queue.h"

ABSL_FLAG(
    bool, reverb_use_workerless_reactors, false,
    "Whether to use workerless reactors. `reverb_insert_worker_num_threads` "
    "and `reverb_sample_worker_num_threads` have no effect when this flag is "
    "set to true.");

// TODO(b/168080187): Benchmark to find good defaults.
ABSL_FLAG(size_t, reverb_insert_worker_num_threads, 32,
          "Number of threads that will run insertion tasks.");
ABSL_FLAG(size_t, reverb_sample_worker_num_threads, 32,
          "Number of threads that will run sample tasks.");
ABSL_FLAG(size_t, reverb_insert_worker_max_queue_size_to_warn, 99000,
          "Size of the queue at which we should warn inserters that the queue "
          "is almost full. Although we use an unbounded queue, inserting too "
          "many items may cause OOMs.");

namespace deepmind {
namespace reverb {
namespace {

// Multiple `ChunkData` can be sent with the same `SampleStreamResponseCtx`. If
// the size of the message exceeds this value then the request is sent and the
// remaining chunks are sent with other messages.
static constexpr int64_t kMaxSampleResponseSizeBytes = 40 * 1024 * 1024;  // 40MB.

inline grpc::Status TableNotFound(absl::string_view name) {
  return grpc::Status(grpc::StatusCode::NOT_FOUND,
                      absl::StrCat("Priority table ", name, " was not found"));
}

inline grpc::Status Internal(const std::string& message) {
  return grpc::Status(grpc::StatusCode::INTERNAL, message);
}

}  // namespace

ReverbServiceAsyncImpl::ReverbServiceAsyncImpl(
    std::shared_ptr<Checkpointer> checkpointer,
    bool use_workerless_reactors)
    : checkpointer_(std::move(checkpointer)),
      use_workerless_reactors_(use_workerless_reactors) {}

absl::Status ReverbServiceAsyncImpl::Create(
    std::vector<std::shared_ptr<Table>> tables,
    std::shared_ptr<Checkpointer> checkpointer,
    std::unique_ptr<ReverbServiceAsyncImpl>* service) {
  // Can't use make_unique because it can't see the Impl's private constructor.
  auto new_service = std::unique_ptr<ReverbServiceAsyncImpl>(
      new ReverbServiceAsyncImpl(std::move(checkpointer),
      absl::GetFlag(FLAGS_reverb_use_workerless_reactors)));
  REVERB_RETURN_IF_ERROR(new_service->Initialize(std::move(tables)));
  std::swap(new_service, *service);
  return absl::OkStatus();
}

absl::Status ReverbServiceAsyncImpl::Create(
    std::vector<std::shared_ptr<Table>> tables,
    std::unique_ptr<ReverbServiceAsyncImpl>* service) {
  return Create(std::move(tables), /*checkpointer=*/nullptr, service);
}

absl::Status ReverbServiceAsyncImpl::Initialize(
    std::vector<std::shared_ptr<Table>> tables) {
  if (checkpointer_ != nullptr) {
    // We start by attempting to load the latest checkpoint from the root
    // directory.
    // In general we expect this to be nonempty (and thus succeed)
    // if this is a restart of a previously running job (e.g preemption).
    auto status = checkpointer_->LoadLatest(&chunk_store_, &tables);
    if (absl::IsNotFound(status)) {
      // No checkpoint was found in the root directory. If a fallback
      // checkpoint (path) has been configured then we attempt to load that
      // checkpoint instead.
      // Note that by first attempting to load from the root directory and
      // then only loading the fallback checkpoint iff the root directory is
      // empty we are effectively using the fallback checkpoint as a way to
      // initialise the service with a checkpoint generated by another
      // experiment.
      status = checkpointer_->LoadFallbackCheckpoint(&chunk_store_, &tables);
    }
    // If no checkpoint was found in neither the root directory nor a fallback
    // checkpoint was provided then proceed to initialise an empty service.
    // All other error types are unexpected and bubbled up to the caller.
    if (!status.ok() && !absl::IsNotFound(status)) {
      return status;
    }
  }

  for (auto& table : tables) {
    std::string name = table->name();
    tables_[name] = std::move(table);
  }

  if (!use_workerless_reactors_) {
    insert_worker_ = absl::make_unique<InsertWorker>(
        absl::GetFlag(FLAGS_reverb_insert_worker_num_threads),
        absl::GetFlag(FLAGS_reverb_insert_worker_max_queue_size_to_warn),
        "InsertWorker");
    sample_worker_ = absl::make_unique<SampleWorker>(
        absl::GetFlag(FLAGS_reverb_sample_worker_num_threads), -1,
        "SampleWorker");
  }

  tables_state_id_ = absl::MakeUint128(absl::Uniform<uint64_t>(rnd_),
                                       absl::Uniform<uint64_t>(rnd_));

  return absl::OkStatus();
}

grpc::ServerUnaryReactor* ReverbServiceAsyncImpl::Checkpoint(
    grpc::CallbackServerContext* context, const CheckpointRequest* request,
    CheckpointResponse* response) {
  grpc::ServerUnaryReactor* reactor = context->DefaultReactor();
  if (checkpointer_ == nullptr) {
    reactor->Finish(
        grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                     "no Checkpointer configured for the replay service."));
    return reactor;
  }

  std::vector<Table*> tables;
  for (auto& table : tables_) {
    tables.push_back(table.second.get());
  }

  auto status = checkpointer_->Save(std::move(tables), 1,
                                    response->mutable_checkpoint_path());
  reactor->Finish(ToGrpcStatus(status));
  REVERB_LOG_IF(REVERB_INFO, status.ok())
      << "Stored checkpoint to " << response->checkpoint_path();
  return reactor;
}


grpc::ServerBidiReactor<InsertStreamRequest, InsertStreamResponse>*
ReverbServiceAsyncImpl::InsertStream(grpc::CallbackServerContext* context) {
  struct InsertStreamResponseCtx {
    InsertStreamResponse payload;
  };

  class InsertReactor
      : public ReverbServerBidiReactor<
            InsertStreamRequest, InsertStreamResponse, InsertStreamResponseCtx,
            InsertTaskInfo, InsertWorker> {
   public:
    InsertReactor(ChunkStore* chunk_store, ReverbServiceAsyncImpl* server)
        : ReverbServerBidiReactor(/*allow_parallel_requests=*/true),
          chunk_store_(chunk_store),
          server_(server) {
      StartRead();
    }

    grpc::Status ProcessIncomingRequest(InsertStreamRequest* request) override {
      REVERB_CHECK(!request->chunks().empty() || request->has_item());

      // ScheduleInsertion will only fail with TableNotFound (i.e., if the
      // item indicates an invalid table name). In that case, we return the
      // error.
      return SaveChunks(request);
    }

    bool ShouldScheduleFirstTask(const InsertStreamRequest& request) override {
      return request.has_item();
    }

    bool ShouldScheduleAnotherTask(const InsertTaskInfo& task_info) override {
      // InsertStream only generates one task per request
      return false;
    }

    grpc::Status FillTaskInfo(InsertStreamRequest* request,
                              InsertTaskInfo* task_info) override {
      if (auto status = GetItemWithChunks(&(task_info->item), request);
          !status.ok()) {
        return status;
      }
      ReleaseOutOfRangeChunks(request->item().keep_chunk_keys());

      const auto& table_name = task_info->item.item.table();

      // Check that table name is valid.
      task_info->table = server_->TableByName(table_name);
      if (task_info->table == nullptr) {
        return TableNotFound(table_name);
      }
      // TODO(b/184716412): Always send confirmations.
      task_info->send_confirmation = request->item().send_confirmation();
      return grpc::Status::OK;
    }

    absl::Status RunTaskAndFillResponses(
        std::vector<InsertStreamResponseCtx>* responses,
        const InsertTaskInfo& task_info) override {
      // We use a (long) timeout here to protect against potential
      // deadlocks were the stream has been closed but we are unable to
      // clean up the reactor due to a indefinetely blocked
      // InsertOrAssign-call.
      REVERB_RETURN_IF_ERROR(
          task_info.table->InsertOrAssign(task_info.item, absl::Seconds(20)));
      if (task_info.send_confirmation) {
        InsertStreamResponseCtx response;
        response.payload.add_keys(task_info.item.item.key());
        responses->push_back(std::move(response));
      }
      return absl::OkStatus();
    }

    InsertWorker* GetWorker() override { return server_->GetInsertWorker(); }

    bool ShouldRetryOnTimeout(const InsertTaskInfo& task_info) override {
      return true;
    }

   private:
    grpc::Status SaveChunks(InsertStreamRequest* request) {
      for (auto& chunk : *request->mutable_chunks()) {
        ChunkStore::Key key = chunk.chunk_key();
        std::shared_ptr<ChunkStore::Chunk> chunk_sp =
            chunk_store_->Insert(std::move(chunk));
        if (chunk_sp == nullptr) {
          return grpc::Status(grpc::StatusCode::CANCELLED,
                              "Service has been closed");
        }
        chunks_[key] = std::move(chunk_sp);
      }

      return grpc::Status::OK;
    }

    grpc::Status GetItemWithChunks(Table::Item* item,
                                   InsertStreamRequest* request) {
      for (ChunkStore::Key key :
           internal::GetChunkKeys(request->item().item().flat_trajectory())) {
        auto it = chunks_.find(key);
        if (it == chunks_.end()) {
          return Internal(
              absl::StrCat("Could not find sequence chunk ", key, "."));
        }
        item->chunks.push_back(it->second);
      }

      item->item = std::move(*request->mutable_item()->mutable_item());

      return grpc::Status::OK;
    }
    void ReleaseOutOfRangeChunks(absl::Span<const uint64_t> keep_keys) {
      for (auto it = chunks_.cbegin(); it != chunks_.cend();) {
        if (std::find(keep_keys.begin(), keep_keys.end(), it->first) ==
            keep_keys.end()) {
          chunks_.erase(it++);
        } else {
          ++it;
        }
      }
      REVERB_CHECK_EQ(chunks_.size(), keep_keys.size())
          << "Kept less chunks than expected.";
    }

    // Incoming messages are handled one at a time. That is StartRead is not
    // called until `request_` has been completely salvaged. Fields accessed
    // only by OnRead are thus thread safe and require no additional mutex to
    // control access.
    //
    // The following fields are ONLY accessed by OnRead (and subcalls):
    //  - chunks_
    //  - chunk_store_

    // Chunks that may be referenced by items not yet received. The ChunkStore
    // itself only maintains weak pointers to the chunk so until an item that
    // references the chunk is created, this pointer is the only reference that
    // stops the chunk from being deallocated.
    internal::flat_hash_map<ChunkStore::Key, std::shared_ptr<ChunkStore::Chunk>>
        chunks_;

    // ChunkStore, only accessed during reads.
    ChunkStore* chunk_store_;

    // Used to lookup tables when inserting items.
    const ReverbServiceAsyncImpl* server_;
  };

  class WorkerlessInsertReactor : public ReverbServerTableReactor<
      InsertStreamRequest, InsertStreamResponse, InsertStreamResponseCtx> {
   public:
    WorkerlessInsertReactor(ChunkStore* chunk_store,
                            ReverbServiceAsyncImpl* server)
        : ReverbServerTableReactor(),
          chunk_store_(chunk_store),
          server_(server),
          continue_inserts_(std::make_shared<std::function<void()>>([&] {
            MaybeStartRead();
        })) {
      MaybeStartRead();
    }

    grpc::Status ProcessIncomingRequest(InsertStreamRequest* request) override {
      REVERB_CHECK(!request->chunks().empty() || request->has_item());
      if (auto status = SaveChunks(request); !status.ok()) {
        return status;
      }
      if (!request->has_item()) {
        // No item to add to the table - continue reading next requests.
        StartRead(&request_);
        return grpc::Status::OK;
      }
      Table::Item item;
      if (auto status = GetItemWithChunks(&item, request); !status.ok()) {
        return status;
      }
      ReleaseOutOfRangeChunks(request->item().keep_chunk_keys());
      const auto& table_name = item.item.table();
      // Check that table name is valid.
      auto table = server_->TableByName(table_name);
      if (table == nullptr) {
        return TableNotFound(table_name);
      }
      bool can_insert;
      const bool send_confirmation = request->item().send_confirmation();
      const auto key = item.item.key();
      if (auto status = table->InsertOrAssignAsync(std::move(item),
          &can_insert, continue_inserts_); !status.ok()) {
        return ToGrpcStatus(status);
      }
      if (can_insert) {
        // Insert didn't exceed table's buffer, we can continue reading next
        // requests.
        StartRead(&request_);
      }
      if (send_confirmation) {
        InsertStreamResponseCtx response;
        response.payload.add_keys(key);
        EnqueueResponse(std::move(response));
      }
      return grpc::Status::OK;
    }

   private:
    grpc::Status SaveChunks(InsertStreamRequest* request) {
      for (auto& chunk : *request->mutable_chunks()) {
        ChunkStore::Key key = chunk.chunk_key();
        std::shared_ptr<ChunkStore::Chunk> chunk_sp =
            chunk_store_->Insert(std::move(chunk));
        if (chunk_sp == nullptr) {
          return grpc::Status(grpc::StatusCode::CANCELLED,
                              "Service has been closed");
        }
        chunks_[key] = std::move(chunk_sp);
      }

      return grpc::Status::OK;
    }

    grpc::Status GetItemWithChunks(Table::Item* item,
                                   InsertStreamRequest* request) {
      for (ChunkStore::Key key :
           internal::GetChunkKeys(request->item().item().flat_trajectory())) {
        auto it = chunks_.find(key);
        if (it == chunks_.end()) {
          return Internal(
              absl::StrCat("Could not find sequence chunk ", key, "."));
        }
        item->chunks.push_back(it->second);
      }

      item->item = std::move(*request->mutable_item()->mutable_item());

      return grpc::Status::OK;
    }

    void ReleaseOutOfRangeChunks(absl::Span<const uint64_t> keep_keys) {
      for (auto it = chunks_.cbegin(); it != chunks_.cend();) {
        if (std::find(keep_keys.begin(), keep_keys.end(), it->first) ==
            keep_keys.end()) {
          chunks_.erase(it++);
        } else {
          ++it;
        }
      }
      REVERB_CHECK_EQ(chunks_.size(), keep_keys.size())
          << "Kept less chunks than expected.";
    }

    // Incoming messages are handled one at a time. That is StartRead is not
    // called until `request_` has been completely salvaged. Fields accessed
    // only by OnRead are thus thread safe and require no additional mutex to
    // control access.
    //
    // The following fields are ONLY accessed by OnRead (and subcalls):
    //  - chunks_
    //  - chunk_store_

    // Chunks that may be referenced by items not yet received. The ChunkStore
    // itself only maintains weak pointers to the chunk so until an item that
    // references the chunk is created, this pointer is the only reference that
    // stops the chunk from being deallocated.
    internal::flat_hash_map<ChunkStore::Key, std::shared_ptr<ChunkStore::Chunk>>
        chunks_;

    // ChunkStore, only accessed during reads.
    ChunkStore* chunk_store_;

    // Used to lookup tables when inserting items.
    const ReverbServiceAsyncImpl* server_;

    // Callback called by the table when further inserts are possible. Pointer
    // to it is registered with the table to avoid memory allocations upon
    // registering callback.
    std::shared_ptr<std::function<void()>> continue_inserts_;
  };

  if (use_workerless_reactors_) {
    return new WorkerlessInsertReactor(&chunk_store_, this);
  } else {
    return new InsertReactor(&chunk_store_, this);
  }
}

grpc::ServerBidiReactor<InitializeConnectionRequest,
                        InitializeConnectionResponse>*
ReverbServiceAsyncImpl::InitializeConnection(
    grpc::CallbackServerContext* context) {
  class Reactor : public grpc::ServerBidiReactor<InitializeConnectionRequest,
                                                 InitializeConnectionResponse> {
   public:
    Reactor(grpc::CallbackServerContext* context,
            ReverbServiceAsyncImpl* server)
        : server_(server) {
      if (!IsLocalhostOrInProcess(context->peer())) {
        Finish(grpc::Status::OK);
        return;
      }

      StartRead(&request_);
    }

    void OnReadDone(bool ok) override {
      if (!ok) {
        Finish(Internal("Failed to read from stream"));
        return;
      }

      if (request_.pid() != getpid()) {
        // A response without an address signal that the client and server are
        // not part of the same process.
        response_.set_address(0);
        StartWrite(&response_);
        return;
      }

      if (table_ptr_ == nullptr) {
        auto table = server_->TableByName(request_.table_name());
        if (table == nullptr) {
          Finish(TableNotFound(request_.table_name()));
          return;
        }

        // Allocate a new shared pointer on the heap and transmit its memory
        // address.
        // The client will dereference and assume ownership of the object before
        // sending its response. For simplicity, the client will copy the
        // shared_ptr so the server is always responsible for cleaning up the
        // heap allocated object.
        table_ptr_ = new std::shared_ptr<Table>(table);

        response_.set_address(reinterpret_cast<int64_t>(table_ptr_));
        StartWrite(&response_);
        return;
      }

      if (!request_.ownership_transferred()) {
        Finish(Internal("Received unexpected request"));
      }

      Finish(grpc::Status::OK);
    }

    void OnWriteDone(bool ok) override {
      if (!ok) {
        Finish(Internal("Failed to write to stream"));
        return;
      }

      // If the address was not set then the client was not running in the same
      // process. No further actions are required so we close down the stream.
      if (response_.address() == 0) {
        Finish(grpc::Status::OK);
        return;
      }

      // Wait for the response from the client confirming that the shared_ptr
      // was copied.
      request_.Clear();
      StartRead(&request_);
    }

    void OnDone() override {
      if (table_ptr_ != nullptr) {
        delete table_ptr_;
      }
      delete this;
    }

   private:
    ReverbServiceAsyncImpl* server_;
    InitializeConnectionRequest request_;
    InitializeConnectionResponse response_;
    std::shared_ptr<Table>* table_ptr_ = nullptr;
  };

  return new Reactor(context, this);
}

grpc::ServerUnaryReactor* ReverbServiceAsyncImpl::MutatePriorities(
    grpc::CallbackServerContext* context,
    const MutatePrioritiesRequest* request,
    MutatePrioritiesResponse* response) {
  grpc::ServerUnaryReactor* reactor = context->DefaultReactor();
  std::shared_ptr<Table> table = TableByName(request->table());
  if (table == nullptr) {
    reactor->Finish(TableNotFound(request->table()));
    return reactor;
  }

  auto status = table->MutateItems(
      std::vector<KeyWithPriority>(request->updates().begin(),
                                   request->updates().end()),
      request->delete_keys());
  reactor->Finish(ToGrpcStatus(status));
  return reactor;
}

grpc::ServerUnaryReactor* ReverbServiceAsyncImpl::Reset(
    grpc::CallbackServerContext* context, const ResetRequest* request,
    ResetResponse* response) {
  grpc::ServerUnaryReactor* reactor = context->DefaultReactor();
  std::shared_ptr<Table> table = TableByName(request->table());
  if (table == nullptr) {
    reactor->Finish(TableNotFound(request->table()));
    return reactor;
  }
  auto status = table->Reset();
  reactor->Finish(ToGrpcStatus(status));
  return reactor;
}

grpc::ServerBidiReactor<SampleStreamRequest, SampleStreamResponse>*
ReverbServiceAsyncImpl::SampleStream(grpc::CallbackServerContext* context) {
  struct SampleStreamResponseCtx {
    SampleStreamResponseCtx(std::shared_ptr<TableItem> table_item)
        : table_item(std::move(table_item)) {
    }
    SampleStreamResponseCtx(const SampleStreamResponseCtx&) = delete;
    SampleStreamResponseCtx& operator=(const SampleStreamResponseCtx&) = delete;
    SampleStreamResponseCtx(SampleStreamResponseCtx&& response) = default;
    SampleStreamResponseCtx& operator=(SampleStreamResponseCtx&& response) =
        default;

    ~SampleStreamResponseCtx() {
      // SampleStreamResponseCtx does not own immutable parts of the payload.
      // We need to make sure not to destroy them while destructing the payload.
      if (payload.info().has_item()) {
        auto item = payload.mutable_info()->mutable_item();
        item->/*unsafe_arena_*/release_inserted_at();
        item->/*unsafe_arena_*/release_flat_trajectory();
      }
      while (payload.data_size() != 0) {
        payload.mutable_data()->UnsafeArenaReleaseLast();
      }
    }

    SampleStreamResponse payload;
    std::shared_ptr<TableItem> table_item;
  };

  class SampleReactor
      : public ReverbServerBidiReactor<
            SampleStreamRequest, SampleStreamResponse, SampleStreamResponseCtx,
            SampleTaskInfo, SampleWorker> {
   public:
    SampleReactor(ReverbServiceAsyncImpl* server)
        : ReverbServerBidiReactor(/*allow_parallel_requests=*/false),
          server_(server) {
      StartRead();
    }

    grpc::Status ProcessIncomingRequest(SampleStreamRequest* request) override {
      if (NumPendingResponses() != 0) {
        return grpc::Status(
            grpc::StatusCode::INTERNAL,
            "Starting a new Sample when the previous one was not completed.");
      }
      fetched_samples_ = 0;
      if (request->num_samples() <= 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            absl::StrCat("`num_samples` must be > 0 (got",
                                         request->num_samples(), ")."));
      }
      if (request->flexible_batch_size() <= 0 &&
          request->flexible_batch_size() != Sampler::kAutoSelectValue) {
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            absl::StrCat("`flexible_batch_size` must be > 0 or ",
                         Sampler::kAutoSelectValue, " (for auto tuning). Got",
                         request->flexible_batch_size(), "."));
      }
      return grpc::Status::OK;
    }

    bool ShouldScheduleFirstTask(const SampleStreamRequest& request) override {
      // The sampler always needs to insert sampling tasks
      return true;
    }

    bool ShouldScheduleAnotherTask(const SampleTaskInfo& task_info) override {
      if (fetched_samples_ == task_info.requested_samples) {
        // It's time to process a new request.
        return false;
      }
      return true;
    }

    grpc::Status FillTaskInfo(SampleStreamRequest* request,
                              SampleTaskInfo* task_info) override {
      if (request->has_rate_limiter_timeout() &&
          request->rate_limiter_timeout().milliseconds() > 0) {
        task_info->retry_on_timeout = false;
        task_info->timeout =
            absl::Milliseconds(request->rate_limiter_timeout().milliseconds());
      } else {
        task_info->retry_on_timeout =
            true;  // this is equivalent to an infinite timeout
        task_info->timeout = absl::Seconds(20);
      }

      task_info->table = server_->TableByName(request->table());
      if (task_info->table == nullptr) {
        return TableNotFound(request->table());
      }
      task_info->flexible_batch_size =
          request->flexible_batch_size() == Sampler::kAutoSelectValue
              ? task_info->table->DefaultFlexibleBatchSize()
              : request->flexible_batch_size();
      task_info->fetched_samples = 0;
      task_info->requested_samples = request->num_samples();
      return grpc::Status::OK;
    }

    SampleTaskInfo FillTaskInfo(const SampleTaskInfo& old_task_info) override {
      return SampleTaskInfo {
        .timeout = old_task_info.timeout,
        .table = old_task_info.table,
        .flexible_batch_size = old_task_info.flexible_batch_size,
        .fetched_samples = fetched_samples_,
        .requested_samples = old_task_info.requested_samples,
        .retry_on_timeout = old_task_info.retry_on_timeout,
      };
    }

    SampleWorker* GetWorker() override { return server_->GetSampleWorker(); }

    absl::Status RunTaskAndFillResponses(
        std::vector<SampleStreamResponseCtx>* responses,
        const SampleTaskInfo& task_info) override {
      std::vector<Table::SampledItem> samples;
      REVERB_RETURN_IF_ERROR(SampleBatch(&samples, task_info));
      // Each sample takes at least one response.
      responses->reserve(responses->size() + samples.size());
      for (auto& sample : samples) {
        EnqueueSample(responses, &sample);
      }
      if (responses->empty()) {
        return absl::InternalError(
            "Sampling a new batch didn't generate new samples.");
      }
      return absl::OkStatus();
    }

    bool ShouldRetryOnTimeout(const SampleTaskInfo& task_info) override {
      return task_info.retry_on_timeout;
    }

   private:
    absl::Status SampleBatch(std::vector<Table::SampledItem>* samples,
                             const SampleTaskInfo& task_info) {
      const int32_t max_batch_size = task_info.NextSampleSize();
      if (auto status = task_info.table->SampleFlexibleBatch(
              samples, max_batch_size, task_info.timeout);
          !status.ok()) {
        return status;
      }
      fetched_samples_ += samples->size();
      return absl::OkStatus();
    }

    void EnqueueSample(std::vector<SampleStreamResponseCtx>* responses,
                       Table::SampledItem* sample) {
      SampleStreamResponseCtx response(sample->ref);
      for (int i = 0; i < sample->ref->chunks.size(); i++) {
        response.payload.set_end_of_sequence(i + 1 ==
                                              sample->ref->chunks.size());

        // Attach the info to the first message.
        if (i == 0) {
          auto* item = response.payload.mutable_info()->mutable_item();
          auto& sample_item = sample->ref->item;
          item->set_key(sample_item.key());
          item->set_table(sample_item.table());
          item->set_priority(sample->priority);
          item->set_times_sampled(sample->times_sampled);
          // ~SampleStreamResponseCtx releases these fields from the proto
          // upon destruction of the item.
          item->/*unsafe_arena_*/set_allocated_inserted_at(
              sample_item.mutable_inserted_at());
          item->/*unsafe_arena_*/set_allocated_flat_trajectory(
              sample_item.mutable_flat_trajectory());
          response.payload.mutable_info()->set_probability(
              sample->probability);
          response.payload.mutable_info()->set_table_size(sample->table_size);
        }

        response.payload.mutable_data()->UnsafeArenaAddAllocated(
            const_cast<ChunkData*>(&sample->ref->chunks[i]->data()));

        if (i < sample->ref->chunks.size() - 1 &&
            response.payload.ByteSizeLong() < kMaxSampleResponseSizeBytes) {
          continue;
        }
        responses->push_back(std::move(response));
        response = SampleStreamResponseCtx(sample->ref);
      }
    }
    // not neeed for thread safety (only one task active, like the chunks stuff)
    int fetched_samples_;
    ReverbServiceAsyncImpl* server_;
  };

  // How often to check whether callback execution finished before deleting
  // reactor.
  static constexpr absl::Duration kCallbackWaitTime = absl::Milliseconds(1);

  class WorkerlessSampleReactor : public ReverbServerTableReactor<
      SampleStreamRequest, SampleStreamResponse, SampleStreamResponseCtx> {
   public:
    using SamplingCallback = std::function<void(Table::SampleRequest*)>;

    WorkerlessSampleReactor(ReverbServiceAsyncImpl* server)
        : ReverbServerTableReactor(),
          server_(server),
          sampling_done_(std::make_shared<SamplingCallback>(
              [&](Table::SampleRequest* sample) {
                if (!sample->status.ok()) {
                  absl::MutexLock lock(&mu_);
                  SetReactorAsFinished(ToGrpcStatus(sample->status));
                  return;
                }
                task_info_.fetched_samples += sample->samples.size();
                for (auto& sample : sample->samples) {
                  ProcessSample(&sample);
                }
                const int next_batch_size = task_info_.NextSampleSize();
                if (next_batch_size == 0) {
                  // Current request is finalized, ask for another one.
                  MaybeStartRead();
                } else {
                  task_info_.table->EnqueSampleRequest(next_batch_size,
                      sampling_done_, task_info_.timeout);
                }
              })) {
      MaybeStartRead();
    }

    ~WorkerlessSampleReactor() {
      // As callback references Reactor's memory make sure it can't be executed
      // anymore.
      std::weak_ptr<SamplingCallback> weak_ptr = sampling_done_;
      sampling_done_.reset();
      while (weak_ptr.lock()) {
        absl::SleepFor(kCallbackWaitTime);
      }
    }

    grpc::Status ProcessIncomingRequest(SampleStreamRequest* request) override {
      if (request->num_samples() <= 0) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            absl::StrCat("`num_samples` must be > 0 (got",
                                         request->num_samples(), ")."));
      }
      if (request->flexible_batch_size() <= 0 &&
          request->flexible_batch_size() != Sampler::kAutoSelectValue) {
        return grpc::Status(
            grpc::StatusCode::INVALID_ARGUMENT,
            absl::StrCat("`flexible_batch_size` must be > 0 or ",
                         Sampler::kAutoSelectValue, " (for auto tuning). Got",
                         request->flexible_batch_size(), "."));
      }
      if (request->has_rate_limiter_timeout() &&
          request->rate_limiter_timeout().milliseconds() > 0) {
        task_info_.timeout =
            absl::Milliseconds(request->rate_limiter_timeout().milliseconds());
      } else {
        task_info_.timeout = absl::InfiniteDuration();
      }

      task_info_.table = server_->TableByName(request->table());
      if (task_info_.table == nullptr) {
        return TableNotFound(request->table());
      }
      task_info_.flexible_batch_size =
          request->flexible_batch_size() == Sampler::kAutoSelectValue
              ? task_info_.table->DefaultFlexibleBatchSize()
              : request->flexible_batch_size();
      task_info_.fetched_samples = 0;
      task_info_.requested_samples = request->num_samples();
      task_info_.table->EnqueSampleRequest(task_info_.NextSampleSize(),
          sampling_done_, task_info_.timeout);
      return grpc::Status::OK;
    }

   private:
    void ProcessSample(Table::SampledItem* sample) {
      SampleStreamResponseCtx response(sample->ref);
      for (int i = 0; i < sample->ref->chunks.size(); i++) {
        response.payload.set_end_of_sequence(i + 1 ==
                                              sample->ref->chunks.size());

        // Attach the info to the first message.
        if (i == 0) {
          auto* item = response.payload.mutable_info()->mutable_item();
          auto& sample_item = sample->ref->item;
          item->set_key(sample_item.key());
          item->set_table(sample_item.table());
          item->set_priority(sample->priority);
          item->set_times_sampled(sample->times_sampled);
          // ~SampleStreamResponseCtx releases these fields from the proto
          // upon destruction of the item.
          item->/*unsafe_arena_*/set_allocated_inserted_at(
              sample_item.mutable_inserted_at());
          item->/*unsafe_arena_*/set_allocated_flat_trajectory(
              sample_item.mutable_flat_trajectory());
          response.payload.mutable_info()->set_probability(
              sample->probability);
          response.payload.mutable_info()->set_table_size(sample->table_size);
        }

        response.payload.mutable_data()->UnsafeArenaAddAllocated(
            const_cast<ChunkData*>(&sample->ref->chunks[i]->data()));

        if (i < sample->ref->chunks.size() - 1 &&
            response.payload.ByteSizeLong() < kMaxSampleResponseSizeBytes) {
          // Response doesn't exceed the size limit yet, so append mode chunks.
          continue;
        }
        // Response exceeds the size limit, so enqueue it for sending to the
        // client and start constructing a new message referencing sampled item.
        // It will contain remaining chunks of the item which didn't fit into
        // the current response.
        {
          absl::MutexLock lock(&mu_);
          EnqueueResponse(std::move(response));
        }
        response = SampleStreamResponseCtx(sample->ref);
      }
    }

    // Used to lookup tables when inserting items.
    const ReverbServiceAsyncImpl* server_;

    // Context of the current sample request.
    SampleTaskInfo task_info_;

    // Callback called by the table worker when current sampling batch is done.
    std::shared_ptr<SamplingCallback> sampling_done_;
  };

  if (use_workerless_reactors_) {
    return new WorkerlessSampleReactor(this);
  } else {
    return new SampleReactor(this);
  }
}

std::shared_ptr<Table> ReverbServiceAsyncImpl::TableByName(
    absl::string_view name) const {
  auto it = tables_.find(name);
  if (it == tables_.end()) return nullptr;
  return it->second;
}

InsertWorker* ReverbServiceAsyncImpl::GetInsertWorker() const {
  return insert_worker_.get();
}

SampleWorker* ReverbServiceAsyncImpl::GetSampleWorker() const {
  return sample_worker_.get();
}

void ReverbServiceAsyncImpl::Close() {
  for (auto& table : tables_) {
    table.second->Close();
  }
}

std::string ReverbServiceAsyncImpl::DebugString() const {
  std::string str = "ReverbServiceAsync(tables=[";
  for (auto iter = tables_.cbegin(); iter != tables_.cend(); ++iter) {
    if (iter != tables_.cbegin()) {
      absl::StrAppend(&str, ", ");
    }
    absl::StrAppend(&str, iter->second->DebugString());
  }
  absl::StrAppend(&str, "], checkpointer=",
                  (checkpointer_ ? checkpointer_->DebugString() : "nullptr"),
                  ")");
  return str;
}

grpc::ServerUnaryReactor* ReverbServiceAsyncImpl::ServerInfo(
    grpc::CallbackServerContext* context, const ServerInfoRequest* request,
    ServerInfoResponse* response) {
  grpc::ServerUnaryReactor* reactor = context->DefaultReactor();
  for (const auto& iter : tables_) {
    *response->add_table_info() = iter.second->info();
  }
  *response->mutable_tables_state_id() = Uint128ToMessage(tables_state_id_);
  reactor->Finish(grpc::Status::OK);
  return reactor;
}

internal::flat_hash_map<std::string, std::shared_ptr<Table>>
ReverbServiceAsyncImpl::tables() const {
  return tables_;
}

}  // namespace reverb
}  // namespace deepmind
