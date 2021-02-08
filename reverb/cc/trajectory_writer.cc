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
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/cleanup.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/support/tf_util.h"
#include "reverb/cc/tensor_compression.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"

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
  request.set_allocated_chunk(const_cast<ChunkData*>(ref.GetChunk().get()));
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

CellRef::CellRef(std::weak_ptr<Chunker> chunker, uint64_t chunk_key, int offset,
                 CellRef::EpisodeInfo episode_info)
    : chunker_(std::move(chunker)),
      chunk_key_(chunk_key),
      offset_(offset),
      episode_info_(std::move(episode_info)),
      chunk_(absl::nullopt) {}

uint64_t CellRef::chunk_key() const { return chunk_key_; }

int CellRef::offset() const { return offset_; }

bool CellRef::IsReady() const {
  absl::MutexLock lock(&mu_);
  return chunk_.has_value();
}

void CellRef::SetChunk(std::shared_ptr<const ChunkData> chunk) {
  absl::MutexLock lock(&mu_);
  chunk_ = std::move(chunk);
}

std::weak_ptr<Chunker> CellRef::chunker() const { return chunker_; }

std::shared_ptr<const ChunkData> CellRef::GetChunk() const {
  absl::MutexLock lock(&mu_);
  return chunk_.value_or(nullptr);
}

uint64_t CellRef::episode_id() const { return episode_info_.episode_id; }

int CellRef::episode_step() const { return episode_info_.step; }

Chunker::Chunker(internal::TensorSpec spec, int max_chunk_length,
                 int num_keep_alive_refs)
    : spec_(std::move(spec)),
      max_chunk_length_(max_chunk_length),
      num_keep_alive_refs_(num_keep_alive_refs) {
  REVERB_CHECK_GE(num_keep_alive_refs, max_chunk_length);
  Reset();
}

absl::Status Chunker::Append(tensorflow::Tensor tensor,
                                   CellRef::EpisodeInfo episode_info,
                                   std::weak_ptr<CellRef>* ref) {
  if (tensor.dtype() != spec_.dtype) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Tensor of wrong dtype provided for column ", spec_.name, ". Got ",
        tensorflow::DataTypeString(tensor.dtype()), " but expected ",
        tensorflow::DataTypeString(spec_.dtype), "."));
  }
  if (!spec_.shape.IsCompatibleWith(tensor.shape())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Tensor of incompatible shape provided for column ", spec_.name,
        ". Got ", tensor.shape().DebugString(), " which is incompatible with ",
        spec_.shape.DebugString(), "."));
  }

  absl::MutexLock lock(&mu_);

  if (!buffer_.empty() &&
      active_refs_.back()->episode_id() != episode_info.episode_id) {
    return absl::FailedPreconditionError(
        "Chunker::Append called with new episode when buffer non empty.");
  }
  if (!buffer_.empty() &&
      active_refs_.back()->episode_step() >= episode_info.step) {
    return absl::FailedPreconditionError(
        "Chunker::Append called with an episode step which was not greater "
        "than already observed.");
  }

  active_refs_.push_back(std::make_shared<CellRef>(
      std::weak_ptr<Chunker>(shared_from_this()), next_chunk_key_, offset_++,
      std::move(episode_info)));

  // Add a batch dim to the tensor before adding it to the buffer. This will
  // prepare it for the concat op when the chunk is finalized.
  tensorflow::TensorShape shape = tensor.shape();
  shape.InsertDim(0, 1);

  // This should never fail due to dtype or shape differences, because the dtype
  // of tensors[j] is UNKNOWN and `shape` has the same number of elements as
  // `item`.
  tensorflow::Tensor batched_tensor(tensor.dtype(), shape);
  REVERB_CHECK(batched_tensor.CopyFrom(tensor, shape));
  buffer_.push_back(std::move(batched_tensor));

  // Create the chunk if max buffer size reached.
  if (buffer_.size() == max_chunk_length_) {
    REVERB_RETURN_IF_ERROR(FlushLocked());
  }

  // Delete references which which have exceeded their max age.
  while (active_refs_.size() > num_keep_alive_refs_) {
    active_refs_.pop_front();
  }

  *ref = active_refs_.back();

  return absl::OkStatus();
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

absl::Status Chunker::Flush() {
  absl::MutexLock lock(&mu_);
  return FlushLocked();
}

absl::Status Chunker::FlushLocked() {
  if (buffer_.empty()) return absl::OkStatus();

  ChunkData chunk;
  chunk.set_chunk_key(next_chunk_key_);

  tensorflow::Tensor batched;
  REVERB_RETURN_IF_ERROR(
      FromTensorflowStatus(tensorflow::tensor::Concat(buffer_, &batched)));
  CompressTensorAsProto(batched, chunk.mutable_data()->add_tensors());

  // Set the sequence range of the chunk.
  for (const auto& ref : active_refs_) {
    if (ref->chunk_key() != chunk.chunk_key()) continue;

    if (!chunk.has_sequence_range()) {
      auto* range = chunk.mutable_sequence_range();
      range->set_episode_id(ref->episode_id());
      range->set_start(ref->episode_step());
      range->set_end(ref->episode_step());
    } else {
      auto* range = chunk.mutable_sequence_range();
      REVERB_CHECK(range->episode_id() == ref->episode_id() &&
                   range->end() < ref->episode_step());

      // The chunk is sparse if not all steps are represented in the data.
      if (ref->episode_step() != range->end() + 1) {
        range->set_sparse(true);
      }
      range->set_end(ref->episode_step());
    }
  }

  // Now the chunk has been finalized we can notify the `CellRef`s.
  auto chunk_sp = std::make_shared<const ChunkData>(std::move(chunk));
  for (auto& ref : active_refs_) {
    if (ref->chunk_key() == chunk_sp->chunk_key()) {
      ref->SetChunk(chunk_sp);
    }
  }

  buffer_.clear();
  next_chunk_key_ = NewKey();
  offset_ = 0;

  return absl::OkStatus();
}

void Chunker::Reset() {
  absl::MutexLock lock(&mu_);
  buffer_.clear();
  buffer_.reserve(max_chunk_length_);
  offset_ = 0;
  next_chunk_key_ = NewKey();
  active_refs_.clear();
}

const internal::TensorSpec& Chunker::spec() const { return spec_; }

absl::Status Chunker::ApplyConfig(int max_chunk_length,
                                  int num_keep_alive_refs) {
  absl::MutexLock lock(&mu_);

  if (!buffer_.empty()) {
    return absl::FailedPreconditionError(
        "Flush must be called before ApplyConfig.");
  }

  TrajectoryWriter::Options options{.max_chunk_length = max_chunk_length,
                                    .num_keep_alive_refs = num_keep_alive_refs};
  REVERB_RETURN_IF_ERROR(options.Validate());

  max_chunk_length_ = max_chunk_length;
  num_keep_alive_refs_ = num_keep_alive_refs;

  while (active_refs_.size() > num_keep_alive_refs) {
    active_refs_.pop_front();
  }

  return absl::OkStatus();
}

TrajectoryWriter::TrajectoryWriter(
    std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
    const Options& options)
    : stub_(std::move(stub)),
      options_(options),
      episode_id_(NewKey()),
      episode_step_(0),
      closed_(false),
      stream_worker_(
          internal::StartThread("TrajectoryWriter_StreamWorker", [this] {
            while (true) {
              auto status = RunStreamWorker();

              absl::MutexLock lock(&mu_);

              if (closed_) {
                unrecoverable_status_ = absl::CancelledError(
                    "TrajectoryWriter::Close has been called.");
                return;
              }

              if (!status.ok() && !absl::IsUnavailable(status)) {
                unrecoverable_status_ = status;
                return;
              }
            }
          })) {
  REVERB_CHECK_OK(options.Validate());
}

TrajectoryWriter::~TrajectoryWriter() {
  {
    absl::MutexLock lock(&mu_);
    if (closed_) return;

    auto status = FlushLocked(/*ignore_last_num_items=*/0,
                              /*timeout=*/absl::InfiniteDuration());
    REVERB_LOG_IF(REVERB_WARNING, !status.ok())
        << "TrajectoryWriter destroyed before content finalized. Encountered "
           "error when trying to finalize content: "
        << status;
  }
  Close();
}

absl::Status TrajectoryWriter::Append(
    std::vector<absl::optional<tensorflow::Tensor>> data,
    std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs) {
  CellRef::EpisodeInfo episode_info;
  {
    absl::MutexLock lock(&mu_);
    REVERB_RETURN_IF_ERROR(unrecoverable_status_);
    episode_info = {episode_id_, episode_step_};
  }

  // If this is the first time the column has been present in the data then
  // create a chunker using the spec of the item.
  for (int i = 0; i < data.size(); i++) {
    if (data[i].has_value() && !chunkers_.contains(i)) {
      const auto& tensor = data[i].value();
      // If the new column has been configured with `ConfigureChunker` then we
      // use the overrided options. If not then we use the default `options_`.
      const auto& chunker_options =
          options_override_.contains(i) ? options_override_[i] : options_;
      chunkers_[i] = std::make_shared<Chunker>(
          internal::TensorSpec{std::to_string(i), tensor.dtype(),
                               tensor.shape()},
          chunker_options.max_chunk_length,
          chunker_options.num_keep_alive_refs);
    }
  }

  // Append data to respective column chunker.
  for (int i = 0; i < data.size(); i++) {
    if (!data[i].has_value()) {
      refs->push_back(absl::nullopt);
      continue;
    }

    std::weak_ptr<CellRef> ref;
    REVERB_RETURN_IF_ERROR(
        chunkers_[i]->Append(std::move(data[i].value()), episode_info, &ref));
    refs->push_back(std::move(ref));
  }

  absl::MutexLock lock(&mu_);

  // Sanity check that `Append` or `EndEpisode` wasn't called concurrently.
  REVERB_CHECK_EQ(episode_info.episode_id, episode_id_);
  REVERB_CHECK_EQ(episode_info.step, episode_step_);

  episode_step_++;

  // Wake up stream worker in case it was blocked on items referencing
  // incomplete chunks
  data_cv_.Signal();

  return absl::OkStatus();
}

absl::Status TrajectoryWriter::InsertItem(
    absl::string_view table, double priority,
    const std::vector<std::vector<std::weak_ptr<CellRef>>>& trajectory) {
  if (trajectory.empty() ||
      std::all_of(trajectory.begin(), trajectory.end(),
                  [](const auto& col) { return col.empty(); })) {
    return absl::InvalidArgumentError("trajectory must not be empty.");
  }

  {
    absl::MutexLock lock(&mu_);
    REVERB_RETURN_IF_ERROR(unrecoverable_status_);
  }

  ItemAndRefs item_and_refs;

  // Lock all the references to ensure that the underlying data is not
  // deallocated before the worker has successfully written the item (and data)
  // to the gRPC stream.
  for (int col_idx = 0; col_idx < trajectory.size(); ++col_idx) {
    const auto& col = trajectory[col_idx];
    if (col.empty()) continue;

    for (auto& ref : col) {
      auto sp = ref.lock();
      if (!sp) {
        return absl::InvalidArgumentError(
            "Trajectory contains expired CellRef.");
      }
      item_and_refs.refs.push_back(std::move(sp));
    }

    // Check that the column only contains compatible data references.
    const auto& col_spec = col[0].lock()->chunker().lock()->spec();
    for (int ref_idx = 1; ref_idx < col.size(); ++ref_idx) {
      const auto& spec = col[ref_idx].lock()->chunker().lock()->spec();
      if (spec.dtype != col_spec.dtype) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Column ", col_idx, " references tensors with different dtypes: ",
            tensorflow::DataTypeString(col_spec.dtype), " (index 0) != ",
            tensorflow::DataTypeString(spec.dtype), " (index ", ref_idx, ")."));
      }
      if (!spec.shape.IsCompatibleWith(col_spec.shape)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Column ", col_idx,
            " references tensors with incompatible shapes: ",
            col_spec.shape.DebugString(), " (index 0) not compatible with ",
            spec.shape.DebugString(), " (index ", ref_idx, ")."));
      }
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

  return absl::OkStatus();
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
    const internal::flat_hash_set<uint64_t>& keep_keys,
    const PrioritizedItem& item) const {
  InsertStreamRequest request;
  request.mutable_item()->set_allocated_item(
      const_cast<PrioritizedItem*>(&item));
  auto realease_item = internal::MakeCleanup(
      [&request] { request.mutable_item()->release_item(); });
  request.mutable_item()->set_send_confirmation(true);
  for (auto keep_key : keep_keys) {
    request.mutable_item()->add_keep_chunk_keys(keep_key);
  }
  return stream->Write(request);
}

internal::flat_hash_set<uint64_t> TrajectoryWriter::GetKeepKeys(
    const internal::flat_hash_set<uint64_t>& streamed_chunk_keys) const {
  internal::flat_hash_set<uint64_t> keys;
  for (const auto& it : chunkers_) {
    for (uint64_t key : it.second->GetKeepKeys()) {
      if (streamed_chunk_keys.contains(key)) {
        keys.insert(key);
      }
    }
  }

  for (auto it = write_queue_.begin(); it != write_queue_.end(); it++) {
    // Ignore chunks only referenced by the front item since keep keys is sent
    // together with this item and thus there is no need for the server to keep
    // these chunks around after the item has been written.
    if (it == write_queue_.begin()) continue;

    for (const auto& ref : it->refs) {
      if (streamed_chunk_keys.contains(ref->chunk_key())) {
        keys.insert(ref->chunk_key());
      }
    }
  }

  return keys;
}

absl::Status TrajectoryWriter::RunStreamWorker() {
  auto stream = SetContextAndCreateStream();

  auto reader = internal::StartThread("TrajectoryWriter_ReaderWorker", [&] {
    InsertStreamResponse response;
    while (stream->Read(&response)) {
      absl::MutexLock lock(&mu_);
      in_flight_items_.erase(response.key());
    }
  });

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

      // Remove keys of expired chunks from streamed_chunk_keys to avoid OOM
      // issues caused by the otherwise indefinitely growing hash set.
      streamed_chunk_keys = GetKeepKeys(streamed_chunk_keys);
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

  return absl::OkStatus();
}

absl::Status TrajectoryWriter::Flush(int ignore_last_num_items,
                                     absl::Duration timeout) {
  absl::MutexLock lock(&mu_);
  return FlushLocked(ignore_last_num_items, timeout);
}

absl::Status TrajectoryWriter::FlushLocked(int ignore_last_num_items,
                                           absl::Duration timeout) {
  // If items are referencing any data which has not yet been finalized into a
  // `ChunkData` then force the chunk to be created prematurely. This will allow
  // the worker to write all items to the stream. Note that we don't need to
  // force the finalization of the `ignore_last_num_items` last items in the
  // queue.
  int num_items_to_force_flush = write_queue_.size() - ignore_last_num_items;
  for (const auto& item : write_queue_) {
    if (num_items_to_force_flush-- <= 0) break;

    for (auto& ref : item.refs) {
      if (!ref->IsReady()) {
        REVERB_RETURN_IF_ERROR(ref->chunker().lock()->Flush());
      }
    }
  }

  // Since all the (referenced) data have been finalized into chunks the worker
  // can be woken up.
  data_cv_.Signal();

  // The write worker is now able to send  (at least) all but the last
  // `ignore_last_num_items` items to the server. We release the mutex and wait
  // for the items to be confirmed or the TrajectoryWriter to be closed.
  auto cond = [ignore_last_num_items,
               this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (!unrecoverable_status_.ok()) {
      return true;
    }

    // Items are considered pending until they are confirmed by the server so
    // both `write_queue_` and `in_flight_items_` must be counted. However, to
    // protect against data races the write worker will wait to add the item to
    // `in_flight_items_` BEFORE removing it from `write_queue_` and then
    // release the mutex for a period in order to perform the actual write to
    // the gRPC stream. We check for this here to avoid double counting.
    int num_pending_items = write_queue_.size() + in_flight_items_.size();
    if (!write_queue_.empty() &&
        in_flight_items_.contains(write_queue_.front().item.key())) {
      num_pending_items--;
    }

    return num_pending_items <= ignore_last_num_items;
  };

  if (!mu_.AwaitWithTimeout(absl::Condition(&cond), timeout)) {
    return absl::DeadlineExceededError(
        absl::StrCat("Timeout exceeded with ", write_queue_.size(),
                     " items waiting to be written and ",
                     in_flight_items_.size(), " items awaiting confirmation."));
  }

  REVERB_RETURN_IF_ERROR(unrecoverable_status_);
  return absl::OkStatus();
}

absl::Status TrajectoryWriter::EndEpisode(bool clear_buffers,
                                                absl::Duration timeout) {
  absl::MutexLock lock(&mu_);
  REVERB_RETURN_IF_ERROR(unrecoverable_status_);

  REVERB_RETURN_IF_ERROR(FlushLocked(0, timeout));

  for (auto& it : chunkers_) {
    if (clear_buffers) {
      it.second->Reset();
    } else {
      // This call should NEVER fail but if it does then we will not be able to
      // recover from it.
      unrecoverable_status_ = it.second->Flush();
      REVERB_RETURN_IF_ERROR(unrecoverable_status_);
    }
  }

  episode_id_ = NewKey();
  episode_step_ = 0;
  return absl::OkStatus();
}

absl::Status TrajectoryWriter::ConfigureChunker(int column,
                                                const Options& options) {
  REVERB_RETURN_IF_ERROR(options.Validate());

  if (auto it = chunkers_.find(column); it != chunkers_.end()) {
    return it->second->ApplyConfig(options.max_chunk_length,
                                   options.num_keep_alive_refs);
  }

  options_override_[column] = options;
  return absl::OkStatus();
}

absl::Status TrajectoryWriter::Options::Validate() const {
  if (max_chunk_length <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "max_chunk_length must be > 0 but got ", max_chunk_length, "."));
  }
  if (num_keep_alive_refs <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "num_keep_alive_refs must be > 0 but got ", num_keep_alive_refs, "."));
  }
  if (max_chunk_length > num_keep_alive_refs) {
    return absl::InvalidArgumentError(absl::StrCat(
        "num_keep_alive_refs (", num_keep_alive_refs,
        ") must be >= max_chunk_length (", max_chunk_length, ")."));
  }
  return absl::OkStatus();
}

}  // namespace reverb
}  // namespace deepmind
