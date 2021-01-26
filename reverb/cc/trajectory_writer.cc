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

#include "reverb/cc/trajectory_writer.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/cleanup.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/tensor_compression.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace deepmind {
namespace reverb {
namespace {

// TODO(b/178091431): Move this into the classes and potentially make it
// injectable so it can be overidden in tests.
uint64_t NewKey() {
  absl::BitGen gen;
  return absl::Uniform<uint64_t>(gen, 0, UINT64_MAX);
}

std::vector<FlatTrajectory::ChunkSlice> MergeAdjacent(
    const std::vector<std::weak_ptr<CellRef>>& refs) {
  std::vector<FlatTrajectory::ChunkSlice> slices;
  for (const auto& ref : refs) {
    // Caller (TrajectoryWriter) is responsible for ensuring that all of the
    // weak pointers are alive.
    auto ref_sp = ref.lock();
    REVERB_CHECK(ref_sp);

    if (slices.empty() || slices.back().chunk_key() != ref_sp->chunk_key()) {
      FlatTrajectory::ChunkSlice slice;
      slice.set_chunk_key(ref_sp->chunk_key());
      slice.set_offset(ref_sp->offset());
      slice.set_length(1);
      slices.push_back(std::move(slice));
    } else {
      slices.back().set_length(slices.back().length() + 1);
    }
  }
  return slices;
}

bool SendChunk(grpc::ClientReaderWriterInterface<InsertStreamRequest,
                                                 InsertStreamResponse>* stream,
               const CellRef& ref) {
  REVERB_CHECK(ref.IsReady());

  InsertStreamRequest request;
  request.set_allocated_chunk(ref.GetChunk().get());
  auto release_chunk =
      internal::MakeCleanup([&request] { request.release_chunk(); });

  grpc::WriteOptions options;
  options.set_no_compression();
  return stream->Write(request, options);
}

bool AllReady(absl::Span<const std::shared_ptr<CellRef>> refs) {
  for (const auto& ref : refs) {
    if (!ref->IsReady()) {
      return false;
    }
  }
  return true;
}

bool ContainsAll(const internal::flat_hash_set<uint64_t>& set,
                 absl::Span<const std::shared_ptr<CellRef>> refs) {
  for (const auto& ref : refs) {
    if (set.find(ref->chunk_key()) == set.end()) {
      return false;
    }
  }
  return true;
}

}  // namespace

CellRef::CellRef(Chunker* chunker, uint64_t chunk_key, int offset)
    : chunker_(chunker),
      chunk_key_(chunk_key),
      offset_(offset),
      chunk_(absl::nullopt) {}

uint64_t CellRef::chunk_key() const { return chunk_key_; }

int CellRef::offset() const { return offset_; }

bool CellRef::IsReady() const {
  absl::MutexLock lock(&mu_);
  return chunk_.has_value();
}

void CellRef::SetChunk(std::shared_ptr<ChunkData> chunk) {
  absl::MutexLock lock(&mu_);
  chunk_ = std::move(chunk);
}

Chunker* CellRef::chunker() const { return chunker_; }

std::shared_ptr<ChunkData> CellRef::GetChunk() const {
  absl::MutexLock lock(&mu_);
  return chunk_.value_or(nullptr);
}

Chunker::Chunker(internal::TensorSpec spec, int max_chunk_length,
                 int num_keep_alive_refs)
    : spec_(std::move(spec)),
      max_chunk_length_(max_chunk_length),
      num_keep_alive_refs_(num_keep_alive_refs),
      offset_(0),
      next_chunk_key_(NewKey()),
      active_refs_() {
  REVERB_CHECK_GE(num_keep_alive_refs, max_chunk_length);
  buffer_.reserve(max_chunk_length);
}

tensorflow::Status Chunker::Append(tensorflow::Tensor tensor,
                                   std::weak_ptr<CellRef>* ref) {
  if (tensor.dtype() != spec_.dtype) {
    return tensorflow::errors::InvalidArgument(
        "Tensor of wrong dtype provided for column ", spec_.name, ". Got ",
        tensorflow::DataTypeString(tensor.dtype()), " but expected ",
        tensorflow::DataTypeString(spec_.dtype), ".");
  }
  if (!spec_.shape.IsCompatibleWith(tensor.shape())) {
    return tensorflow::errors::InvalidArgument(
        "Tensor of incompatible shape provided for column ", spec_.name,
        ". Got ", tensor.shape().DebugString(), " which is incompatible with ",
        spec_.shape.DebugString(), ".");
  }

  absl::MutexLock lock(&mu_);

  active_refs_.push_back(
      std::make_shared<CellRef>(this, next_chunk_key_, offset_++));
  buffer_.push_back(std::move(tensor));

  // Create the chunk if max buffer size reached.
  if (buffer_.size() == max_chunk_length_) {
    TF_RETURN_IF_ERROR(FlushLocked());
  }

  // Delete references which which have exceeded their max age.
  while (active_refs_.size() > num_keep_alive_refs_) {
    active_refs_.pop_front();
  }

  *ref = active_refs_.back();

  return tensorflow::Status::OK();
}

std::vector<uint64_t> Chunker::GetKeepKeys() const {
  absl::MutexLock lock(&mu_);
  std::vector<uint64_t> keys;
  for (const auto& ref : active_refs_) {
    if (keys.empty() || keys.back() != ref->chunk_key()) {
      keys.push_back(ref->chunk_key());
    }
  }
  return keys;
}

tensorflow::Status Chunker::Flush() {
  absl::MutexLock lock(&mu_);
  return FlushLocked();
}

tensorflow::Status Chunker::FlushLocked() {
  if (buffer_.empty()) return tensorflow::Status::OK();

  auto chunk = std::make_shared<ChunkData>();
  chunk->set_chunk_key(next_chunk_key_);

  tensorflow::Tensor batched;
  TF_RETURN_IF_ERROR(tensorflow::tensor::Concat(buffer_, &batched));
  CompressTensorAsProto(batched, chunk->mutable_data()->add_tensors());

  for (auto& ref : active_refs_) {
    if (ref->chunk_key() == chunk->chunk_key()) {
      ref->SetChunk(chunk);
    }
  }

  buffer_.clear();
  next_chunk_key_ = NewKey();
  offset_ = 0;

  return tensorflow::Status::OK();
}

TrajectoryWriter::TrajectoryWriter(
    std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
    const Options& options)
    : stub_(std::move(stub)),
      options_(options),
      closed_(false),
      stream_worker_(
          internal::StartThread("TrajectoryWriter_StreamWorker", [this] {
            while (true) {
              auto status = RunStreamWorker();

              absl::MutexLock lock(&mu_);

              if (closed_) {
                unrecoverable_status_ = tensorflow::errors::Cancelled(
                    "TrajectoryWriter::Close has been called.");
                return;
              }

              if (!status.ok() && !tensorflow::errors::IsUnavailable(status)) {
                unrecoverable_status_ = status;
                return;
              }
            }
          })) {}

TrajectoryWriter::~TrajectoryWriter() {
  {
    absl::MutexLock lock(&mu_);
    if (closed_) return;
  }

  auto status = Flush();
  REVERB_LOG_IF(REVERB_WARNING, !status.ok())
      << "TrajectoryWriter destroyed before content finalized. Encountered "
         "error when trying to finalize content: "
      << status;

  Close();
}

tensorflow::Status TrajectoryWriter::Append(
    std::vector<absl::optional<tensorflow::Tensor>> data,
    std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs) {
  {
    absl::MutexLock lock(&mu_);
    TF_RETURN_IF_ERROR(unrecoverable_status_);
  }

  // If this is the first time the column has been present in the data then
  // create a chunker using the spec of the item.
  for (int i = 0; i < data.size(); i++) {
    if (data[i].has_value() && !chunkers_.contains(i)) {
      const auto& tensor = data[i].value();
      chunkers_[i] = absl::make_unique<Chunker>(
          internal::TensorSpec{std::to_string(i), tensor.dtype(),
                               tensor.shape()},
          options_.max_chunk_length, options_.num_keep_alive_refs);
    }
  }

  // Append data to respective column chunker.
  for (int i = 0; i < data.size(); i++) {
    if (!data[i].has_value()) {
      refs->push_back(absl::nullopt);
      continue;
    }

    std::weak_ptr<CellRef> ref;
    TF_RETURN_IF_ERROR(chunkers_[i]->Append(std::move(data[i].value()), &ref));
    refs->push_back(std::move(ref));
  }

  // Wake up stream worker in case it was blocked on items referencing
  // incomplete chunks.
  {
    absl::MutexLock lock(&mu_);
    data_cv_.Signal();
  }

  return tensorflow::Status::OK();
}

tensorflow::Status TrajectoryWriter::InsertItem(
    absl::string_view table, double priority,
    const std::vector<std::vector<std::weak_ptr<CellRef>>>& trajectory) {
  {
    absl::MutexLock lock(&mu_);
    TF_RETURN_IF_ERROR(unrecoverable_status_);
  }

  ItemAndRefs item_and_refs;

  // Lock all the references to ensure that the underlying data is not
  // deallocated before the worker has successfully written the item (and data)
  // to the gRPC stream.
  for (const auto& col : trajectory) {
    for (auto& ref : col) {
      auto sp = ref.lock();
      if (!sp) {
        return tensorflow::errors::InvalidArgument(
            "Trajectory contains expired CellRef.");
      }
      item_and_refs.refs.push_back(std::move(sp));
    }
  }

  item_and_refs.item.set_key(NewKey());
  item_and_refs.item.set_table(table.data(), table.size());
  item_and_refs.item.set_priority(priority);

  for (const auto& column_refs : trajectory) {
    auto* col = item_and_refs.item.mutable_flat_trajectory()->add_columns();
    // Note that MergeAdjacent can safely assume that all weak_ptrs are alive
    // since the corresponding shared_ptrs exists in item_and_refs.
    for (auto& slice : MergeAdjacent(column_refs)) {
      *col->add_chunk_slices() = std::move(slice);
    }
  }

  {
    absl::MutexLock lock(&mu_);
    write_queue_.push_back(std::move(item_and_refs));
  }

  return tensorflow::Status::OK();
}

void TrajectoryWriter::Close() {
  {
    absl::MutexLock lock(&mu_);
    if (closed_) return;

    // This will unblock the worker if it is waiting for new items to be sent.
    closed_ = true;

    // This will unblock the worker if it is blocked on a Write-call. It will
    // also unblock the reader thread as it definetely is blocked on a
    // Read-call.
    if (context_ != nullptr) {
      context_->TryCancel();
    }

    // This will unblock the worker if the front pending item is referencing
    // incomplete chunks and the worker is waiting for that to change.
    data_cv_.Signal();
  }

  // Join the worker thread.
  stream_worker_ = nullptr;
}

std::unique_ptr<TrajectoryWriter::InsertStream>
TrajectoryWriter::SetContextAndCreateStream() {
  absl::MutexLock lock(&mu_);
  REVERB_CHECK(unrecoverable_status_.ok());
  context_ = absl::make_unique<grpc::ClientContext>();
  context_->set_wait_for_ready(false);
  return stub_->InsertStream(context_.get());
}

bool TrajectoryWriter::GetNextPendingItem(
    TrajectoryWriter::ItemAndRefs* item_and_refs) const {
  absl::MutexLock lock(&mu_);
  mu_.Await(absl::Condition(
      +[](const TrajectoryWriter* w) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        return !w->write_queue_.empty() || w->closed_;
      },
      this));

  if (closed_) return false;

  *item_and_refs = write_queue_.front();

  return true;
}

bool TrajectoryWriter::SendItem(
    TrajectoryWriter::InsertStream* stream,
    const internal::flat_hash_set<uint64_t>& streamed_chunk_keys,
    const PrioritizedItem& item) const {
  InsertStreamRequest request;
  request.mutable_item()->set_allocated_item(
      const_cast<PrioritizedItem*>(&item));
  auto realease_item = internal::MakeCleanup(
      [&request] { request.mutable_item()->release_item(); });
  request.mutable_item()->set_send_confirmation(true);
  for (const auto& it : chunkers_) {
    for (auto key : it.second->GetKeepKeys()) {
      if (streamed_chunk_keys.contains(key)) {
        request.mutable_item()->add_keep_chunk_keys(key);
      }
    }
  }
  return stream->Write(request);
}

tensorflow::Status TrajectoryWriter::RunStreamWorker() {
  auto stream = SetContextAndCreateStream();

  auto reader = internal::StartThread("TrajectoryWriter_ReaderWorker", [&] {
    InsertStreamResponse response;
    while (stream->Read(&response)) {
      absl::MutexLock lock(&mu_);
      in_flight_items_.erase(response.key());
    }
  });

  // TODO(b/178090180): This will continue to grow indef. It is a very small
  // memory leak but if the writer is kept alive for a long time then it could
  // become a problem.
  internal::flat_hash_set<uint64_t> streamed_chunk_keys;
  while (true) {
    ItemAndRefs item_and_refs;

    if (!GetNextPendingItem(&item_and_refs)) {
      return FromGrpcStatus(stream->Finish());
    }

    // Send referenced chunks which haven't already been sent.
    for (const auto& ref : item_and_refs.refs) {
      if (!ref->IsReady() || streamed_chunk_keys.contains(ref->chunk_key())) {
        continue;
      }
      if (!SendChunk(stream.get(), *ref)) {
        return FromGrpcStatus(stream->Finish());
      }
      streamed_chunk_keys.insert(ref->chunk_key());
    }

    {
      absl::WriterMutexLock lock(&mu_);
      // Check whether all chunks referenced by the item have been written to
      // the stream. If not, then at least one chunk is incomplete and the
      // worker will wait for the chunk state to change and then retry.
      if (!ContainsAll(streamed_chunk_keys, item_and_refs.refs)) {
        // Do a final check that the chunks didn't change since the lock was
        // last held. If the item still references incomplete chunks then we
        // sleep until the chunks changed. If all the chunks are now completed
        // then we move straight to the top of the loop.
        if (!AllReady(item_and_refs.refs)) {
          data_cv_.Wait(&mu_);
        }
        continue;
      }

      in_flight_items_.insert(item_and_refs.item.key());
    }

    // All chunks have been written to the stream so the item can now be
    // written.
    if (!SendItem(stream.get(), streamed_chunk_keys, item_and_refs.item)) {
      {
        absl::WriterMutexLock lock(&mu_);
        in_flight_items_.erase(item_and_refs.item.key());
      }
      return FromGrpcStatus(stream->Finish());
    }

    // Item has been sent so we can now pop it from the queue. Note that by
    // deallocating the ItemAndRefs object we are allowing the underlying
    // ChunkData to be deallocated.
    {
      absl::MutexLock lock(&mu_);
      // TODO(b/178090185): Maybe keep the item and references around until the
      // item has been confirmed by the server.
      write_queue_.pop_front();
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status TrajectoryWriter::Flush(absl::Duration timeout) {
  absl::MutexLock lock(&mu_);

  // If items are referencing any data which has not yet been finalized into a
  // `ChunkData` then force the chunk to be created prematurely. This will allow
  // the worker to write all items to the stream.
  for (const auto& item : write_queue_) {
    for (auto& ref : item.refs) {
      if (!ref->IsReady()) {
        TF_RETURN_IF_ERROR(ref->chunker()->Flush());
      }
    }
  }

  // Since all the (referenced) data have been finalized into chunks the worker
  // can be woken up.
  data_cv_.Signal();

  // The write worker is now able to send all pending items so we can now await
  // all in flight items to be empty or for the TrajectoryWriter to be closed.
  if (!mu_.AwaitWithTimeout(
          absl::Condition(
              +[](TrajectoryWriter* w) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                return (w->write_queue_.empty() &&
                        w->in_flight_items_.empty()) ||
                       !w->unrecoverable_status_.ok();
              },
              this),
          timeout)) {
    return tensorflow::errors::DeadlineExceeded(
        "Timeout exceeded with ", write_queue_.size(),
        " items waiting to be written and ", in_flight_items_.size(),
        " items awaiting confirmation.");
  }

  TF_RETURN_IF_ERROR(unrecoverable_status_);
  return tensorflow::Status::OK();
}

}  // namespace reverb
}  // namespace deepmind
