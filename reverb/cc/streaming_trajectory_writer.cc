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

#include "reverb/cc/streaming_trajectory_writer.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/key_generators.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/trajectory_writer.h"
#include "tensorflow/core/framework/tensor.h"

namespace deepmind::reverb {
namespace {

// We can recover from transient errors by starting a new stream.
bool IsTransientError(const absl::Status& status) {
  return absl::IsDeadlineExceeded(status) || absl::IsUnavailable(status) ||
         absl::IsCancelled(status);
}

// Clears the vector on destruction unless `set_clear(false)` is called. Can be
// used to ensure vectors returned via argument pointers only hold values if the
// function succeeds entirely.
template <typename T>
class ClearVectorOnExit {
 public:
  explicit ClearVectorOnExit(T vec) : vec_(vec) {}
  ~ClearVectorOnExit() {
    if (clear_) vec_->clear();
  }

  // Call this function before destruction to avoid clearing the vector.
  void do_not_clear() { clear_ = false; }

 private:
  T vec_;
  bool clear_ = true;
};

}  // namespace

StreamingTrajectoryWriter::StreamingTrajectoryWriter(
    std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
    const TrajectoryWriter::Options& options)
    : stub_(std::move(stub)),
      options_(options),
      episode_id_(key_generator_.Generate()),
      episode_step_(0),
      unrecoverable_error_(absl::OkStatus()),
      recoverable_error_(absl::OkStatus()) {
  REVERB_CHECK(options_.chunker_options != nullptr);
  REVERB_CHECK_OK(options.Validate());
  SetContextAndCreateStream();
}

StreamingTrajectoryWriter::~StreamingTrajectoryWriter() {
  // Make sure to flush the stream on destruction.
  if (stream_) {
    stream_->WritesDone();
    absl::Status status = FromGrpcStatus(stream_->Finish());
    REVERB_LOG_IF(REVERB_ERROR, !status.ok())
        << "Failed to close stream: " << status;
    item_confirmation_worker_ = nullptr;  // Join thread.
  }
}

absl::Status StreamingTrajectoryWriter::Append(
    std::vector<absl::optional<tensorflow::Tensor>> data,
    std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs) {
  return AppendInternal(std::move(data), /*increment_episode_step=*/true, refs);
}

absl::Status StreamingTrajectoryWriter::AppendPartial(
    std::vector<absl::optional<tensorflow::Tensor>> data,
    std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs) {
  return AppendInternal(std::move(data), /*increment_episode_step=*/false,
                        refs);
}

absl::Status StreamingTrajectoryWriter::AppendInternal(
    std::vector<absl::optional<tensorflow::Tensor>> data,
    bool increment_episode_step,
    std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs) {
  REVERB_CHECK(refs != nullptr);
  ClearVectorOnExit<decltype(refs)> clear(refs);
  REVERB_RETURN_IF_ERROR(unrecoverable_error_);
  REVERB_RETURN_IF_ERROR(recoverable_error_);

  CellRef::EpisodeInfo episode_info{episode_id_, episode_step_};

  // If this is the first time the column has been present in the data then
  // create a chunker using the spec of the item.
  for (int i = 0; i < data.size(); i++) {
    if (data[i].has_value() && !chunkers_.contains(i)) {
      const tensorflow::Tensor& tensor = data[i].value();
      chunkers_[i] = std::make_shared<Chunker>(
          internal::TensorSpec{std::to_string(i), tensor.dtype(),
                               tensor.shape()},
          options_.chunker_options->Clone());
    }
  }

  // Collect references to stream completed chunks.
  std::vector<std::shared_ptr<CellRef>> refs_sps;

  // Append data to respective column chunker.
  for (int i = 0; i < data.size(); i++) {
    if (!data[i].has_value()) {
      refs->push_back(absl::nullopt);
      continue;
    }

    std::weak_ptr<CellRef> ref;
    absl::Status status =
        chunkers_[i]->Append(data[i].value(), episode_info, &ref);
    if (absl::IsFailedPrecondition(status)) {
      return absl::FailedPreconditionError(
          "Append/AppendPartial called with data containing column that was "
          "present in previous AppendPartial call.");
    }
    REVERB_RETURN_IF_ERROR(status);

    refs_sps.push_back(ref.lock());
    refs->push_back(std::move(ref));
  }

  if (increment_episode_step) ++episode_step_;

  absl::Status result = StreamChunks(refs_sps);

  // If everything went ok, we don't need to clear the vector.
  if (result.ok()) clear.do_not_clear();

  return result;
}

absl::Status StreamingTrajectoryWriter::CreateItem(
    absl::string_view table, double priority,
    absl::Span<const TrajectoryColumn> trajectory) {
  REVERB_RETURN_IF_ERROR(unrecoverable_error_);
  REVERB_RETURN_IF_ERROR(recoverable_error_);

  if (trajectory.empty() ||
      std::all_of(trajectory.begin(), trajectory.end(),
                  [](const TrajectoryColumn& col) { return col.empty(); })) {
    return absl::InvalidArgumentError("trajectory must not be empty.");
  }

  TrajectoryWriter::ItemAndRefs item_and_refs;

  // Lock all the references to ensure that the underlying data is not
  // deallocated before the worker has successfully written the item (and data)
  // to the gRPC stream.
  for (int col_idx = 0; col_idx < trajectory.size(); ++col_idx) {
    if (absl::Status status = trajectory[col_idx].Validate(); !status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Error in column ", col_idx, ": ", status.message()));
    }
    if (!trajectory[col_idx].LockReferences(&item_and_refs.refs)) {
      return absl::InternalError("CellRef unexpectedly expired in CreateItem.");
    }
  }

  // All chunks referenced by this item must have been streamed to the replay
  // buffer prior to inserting the item. Finalize chunks that aren't ready.
  for (std::shared_ptr<CellRef>& ref : item_and_refs.refs) {
    if (!ref->IsReady()) {
      REVERB_RETURN_IF_ERROR(ref->chunker().lock()->Flush());
    }
  }
  REVERB_RETURN_IF_ERROR(StreamChunks(item_and_refs.refs));

  item_and_refs.item.set_key(key_generator_.Generate());
  item_and_refs.item.set_table(table.data(), table.size());
  item_and_refs.item.set_priority(priority);

  for (const TrajectoryColumn& column : trajectory) {
    column.ToProto(item_and_refs.item.mutable_flat_trajectory()->add_columns());
  }

  REVERB_RETURN_IF_ERROR(item_and_refs.Validate(options_));

  for (auto& [_, chunker] : chunkers_) {
    chunker->OnItemFinalized(item_and_refs.item, item_and_refs.refs);
  }

  // All chunks have been written to the stream so the item can now be
  // written.
  return SendItem(std::move(item_and_refs.item));
}

absl::Status StreamingTrajectoryWriter::EndEpisode(bool clear_buffers,
                                                   absl::Duration timeout) {
  REVERB_RETURN_IF_ERROR(unrecoverable_error_);

  // Wait for all items belonging to this episode to be confirmed. This only
  // makes sense if the stream hasn't failed.
  if (recoverable_error_.ok()) {
    REVERB_RETURN_IF_ERROR(Flush(0, timeout));
  }

  episode_id_ = key_generator_.Generate();
  episode_step_ = 0;

  if (clear_buffers) {
    streamed_chunk_keys_.clear();
    recoverable_error_ = absl::OkStatus();
    for (auto& [_, chunker] : chunkers_) {
      chunker->Reset();
    }
  }

  return absl::OkStatus();
}

absl::Status StreamingTrajectoryWriter::Flush(int ignore_last_num_items,
                                              absl::Duration timeout) {
  absl::MutexLock lock(&mutex_);

  // We assume that confirmations arrive in order of insertion. This means if
  // in_flight_items_.size() is less than or equal to N, all but the last N
  // items have been confirmed. If confirmations arrive out-of-order, we'd wait
  // until all but N (instead of the last N) items are confirmed.
  auto condition = [ignore_last_num_items, this]()
                       ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) -> bool {
    if (!unrecoverable_error_.ok() || !recoverable_error_.ok()) return true;

    return in_flight_items_.size() <= ignore_last_num_items;
  };

  if (!mutex_.AwaitWithTimeout(absl::Condition(&condition), timeout)) {
    return absl::DeadlineExceededError(
        absl::StrCat("Timeout exceeded with ", in_flight_items_.size(),
                     " items awaiting confirmation."));
  }

  REVERB_RETURN_IF_ERROR(unrecoverable_error_);
  return recoverable_error_;
}

absl::Status StreamingTrajectoryWriter::StreamChunks(
    const std::vector<std::shared_ptr<CellRef>>& refs) {
  // We store requests in a vector as we may shard chunk insertions over several
  // requests to improve performance.
  std::vector<InsertStreamRequest> requests(1);

  std::vector<uint64_t> chunk_keys;

  for (const std::shared_ptr<CellRef>& ref : refs) {
    // Skip this chunk if it's not ready yet.
    if (!ref->IsReady()) continue;

    // Take over ownership of the chunk.
    std::unique_ptr<const ChunkData> chunk = std::move(ref->GetChunk()->chunk);

    // If the chunk was streamed previously, we must find it in the list of
    // streamed chunk keys.
    if (chunk == nullptr) {
      if (!streamed_chunk_keys_.contains(ref->chunk_key()) &&
          absl::c_find(chunk_keys, ref->chunk_key()) == chunk_keys.end()) {
        return absl::InvalidArgumentError(
            "Chunk key of previously streamed chunk does not belong to this "
            "episode.");
      }
      continue;
    }

    chunk_keys.push_back(chunk->chunk_key());
    // Pass ownership of the chunk to the request proto.
    requests.back().mutable_chunks()->AddAllocated(
        const_cast<ChunkData*>(chunk.release()));

    // If the size of the request exceeds the soft threshold, shard the
    // insertion and start a new request.
    if (requests.back().ByteSizeLong() >=
        StreamingTrajectoryWriter::kMaxRequestSizeBytes) {
      requests.emplace_back();
    }
  }

  if (chunk_keys.empty()) return absl::OkStatus();

  // Send all requests.
  for (const InsertStreamRequest& request : requests) {
    REVERB_RETURN_IF_ERROR(WriteStream(request));
  }

  streamed_chunk_keys_.insert(chunk_keys.begin(), chunk_keys.end());

  return absl::OkStatus();
}

absl::Status StreamingTrajectoryWriter::SendItem(PrioritizedItem item) {
  InsertStreamRequest request;

  *request.mutable_item()->mutable_item() = std::move(item);
  request.mutable_item()->set_send_confirmation(true);

  // Keep all chunk keys belonging to this episode since we don't know which
  // chunks that aren't referenced by any item at the moment will be needed by
  // the next item.
  for (uint64_t keep_key : streamed_chunk_keys_) {
    request.mutable_item()->add_keep_chunk_keys(keep_key);
  }

  return WriteStream(request);
}

absl::Status StreamingTrajectoryWriter::WriteStream(
    const InsertStreamRequest& request) {
  grpc::WriteOptions options;
  options.set_no_compression();

  // If this request contains an item, mark it as "in-flight".
  if (request.has_item()) {
    absl::MutexLock lock(&mutex_);
    in_flight_items_.insert(request.item().item().key());
  }

  if (!stream_->Write(request, options)) {
    // We won't get a confirmation for this item.
    if (request.has_item()) {
      absl::MutexLock lock(&mutex_);
      in_flight_items_.erase(request.item().item().key());
    }

    // We can recover from transient errors as they only corrupt the current
    // episode.
    absl::Status streaming_status = FromGrpcStatus(stream_->Finish());

    // Join the confirmation thread.
    item_confirmation_worker_ = nullptr;

    if (IsTransientError(streaming_status)) {
      SetContextAndCreateStream();
      recoverable_error_ = absl::DataLossError(absl::StrCat(
          "Stream interrupted with error: ", streaming_status.message()));
      return recoverable_error_;
    } else {
      unrecoverable_error_ = streaming_status;
      return unrecoverable_error_;
    }
  }

  return absl::OkStatus();
}

void StreamingTrajectoryWriter::ProcessItemConfirmations() {
  InsertStreamResponse response;
  while (stream_->Read(&response)) {
    absl::MutexLock lock(&mutex_);
    for (uint64_t key : response.keys()) {
      in_flight_items_.erase(key);
    }
    response.Clear();
  }
}

void StreamingTrajectoryWriter::SetContextAndCreateStream() {
  context_ = absl::make_unique<grpc::ClientContext>();
  context_->set_wait_for_ready(true);
  stream_ = stub_->InsertStream(context_.get());
  item_confirmation_worker_ =
      internal::StartThread("StreamingTrajectoryWriter_ReaderWorker",
                            [this]() { ProcessItemConfirmations(); });
}

}  // namespace deepmind::reverb
