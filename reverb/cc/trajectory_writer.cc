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

#include <limits>
#include <memory>
#include <vector>

#include "grpcpp/impl/codegen/sync_stream.h"
#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/cleanup.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/key_generators.h"
#include "reverb/cc/support/trajectory_util.h"
#include "tensorflow/core/framework/types.h"

namespace deepmind {
namespace reverb {

class ArenaOwnedRequest {
 public:
  ~ArenaOwnedRequest() { Clear(); }

  void Clear() {
    while (!r.chunks().empty()) {
      r.mutable_chunks()->UnsafeArenaReleaseLast();
    }
    if (r.has_item()) {
      r.mutable_item()->unsafe_arena_release_item();
      r.clear_item();
    }
  }
  InsertStreamRequest r;
};

namespace {

std::vector<FlatTrajectory::ChunkSlice> MergeAdjacent(
    const std::vector<std::weak_ptr<CellRef>>& refs) {
  std::vector<FlatTrajectory::ChunkSlice> slices;
  for (const std::weak_ptr<CellRef>& ref : refs) {
    // Caller (TrajectoryWriter) is responsible for ensuring that all of the
    // weak pointers are alive.
    std::shared_ptr<CellRef> ref_sp = ref.lock();
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


// Returns true if all references `refs` are ready.
bool AllReady(absl::Span<const std::shared_ptr<CellRef>> refs) {
  return absl::c_all_of(refs, [](const auto& ref) { return ref->IsReady(); });
}

// Returns true if `set` contains all chunk keys references by `refs`.
bool ContainsAll(const internal::flat_hash_set<uint64_t>& set,
                 absl::Span<const std::shared_ptr<CellRef>> refs) {
  return absl::c_all_of(refs, [&set](const auto& ref) {
    return set.template contains(ref->chunk_key());
  });
}

std::vector<internal::TensorSpec> FlatSignatureFromTrajectory(
    const FlatTrajectory& trajectory,
    absl::Span<const std::shared_ptr<CellRef>> refs) {
  auto get_spec = [&](uint64_t chunk_key) -> internal::TensorSpec {
    for (const auto& ref : refs) {
      if (ref->chunk_key() == chunk_key) {
        return ref->chunker().lock()->spec();
      }
    }
    REVERB_CHECK(false) << "Invalid trajectory";
  };

  std::vector<internal::TensorSpec> specs;
  for (int col_idx = 0; col_idx < trajectory.columns_size(); col_idx++) {
    const FlatTrajectory::Column& col = trajectory.columns(col_idx);
    internal::TensorSpec spec = get_spec(col.chunk_slices(0).chunk_key());
    spec.name = std::to_string(col_idx);
    if (!col.squeeze()) {
      spec.shape.InsertDim(0, internal::ColumnLength(trajectory, col_idx));
    }
    specs.push_back(std::move(spec));
  }
  return specs;
}

}  // namespace

bool TrajectoryWriter::Write(ArenaOwnedRequest* request) {
  {
    absl::MutexLock lock(&mu_);
    write_inflight_ = true;
  }
  grpc::WriteOptions options;
  options.set_no_compression();
  StartWrite(&request->r, options);
  {
    absl::MutexLock lock(&mu_);
    auto trigger = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return !write_inflight_ || closed_ || !stream_ok_;
    };
    mu_.Await(absl::Condition(&trigger));
    request->Clear();
    return !write_inflight_;
  }
}

bool TrajectoryWriter::SendNotAlreadySentChunks(
    internal::flat_hash_set<uint64_t>* streamed_chunk_keys,
    absl::Span<const std::shared_ptr<CellRef>> refs,
    ArenaOwnedRequest* request) {
  // Send referenced chunks which haven't already been sent.
  for (const std::shared_ptr<CellRef>& ref : refs) {
    if (!ref->IsReady() || streamed_chunk_keys->contains(ref->chunk_key())) {
      continue;
    }
    request->r.mutable_chunks()->UnsafeArenaAddAllocated(
        const_cast<ChunkData*>(ref->GetChunk()->get()));
    streamed_chunk_keys->insert(ref->chunk_key());

    // If the message has grown beyond the cutoff point then we send it.
    if (request->r.ByteSizeLong() >= TrajectoryWriter::kMaxRequestSizeBytes) {
      if (!Write(request)) {
        return false;
      }

      // There (might) still be chunks which can be transmitted so continue with
      // the remaining references.
    }
  }
  // Remaining chunks will be sent together with the Item.
  return true;
}

absl::Status TrajectoryWriter::Options::Validate() const {
  if (chunker_options == nullptr) {
    return absl::InvalidArgumentError("chunker_options must be set.");
  }
  return ValidateChunkerOptions(chunker_options.get());
}

absl::Status TrajectoryWriter::ItemAndRefs::Validate(
    const TrajectoryWriter::Options& options) const {
  if (!options.flat_signature_map.has_value()) {
    return absl::OkStatus();
  }

  const std::string& table = item.table();
  const internal::FlatSignatureMap& signature_map =
      options.flat_signature_map.value();
  auto it = signature_map.find(table);
  if (it == signature_map.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unable to create item in table '%s' since the table "
                        "could not be found.",
                        table));
  }
  if (!it->second.has_value()) {
    return absl::OkStatus();
  }
  const std::vector<internal::TensorSpec>& table_signature = it->second.value();

  const FlatTrajectory& trajectory = item.flat_trajectory();
  std::vector<internal::TensorSpec> trajectory_signature =
      FlatSignatureFromTrajectory(trajectory, refs);

  if (table_signature.size() != trajectory_signature.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unable to create item in table '%s' since the provided trajectory "
        "is inconsistent with the table signature. The trajectory has %d "
        "columns but the table signature has %d columns."
        "\n\nThe table signature is:\n\t%s"
        "\n\nThe provided trajectory signature was:\n\t%s.\n",
        table, trajectory_signature.size(), table_signature.size(),
        internal::DtypesShapesString(table_signature),
        internal::DtypesShapesString(trajectory_signature)));
  }

  for (int i = 0; i < table_signature.size(); i++) {
    const internal::TensorSpec& want = table_signature[i];
    const internal::TensorSpec& got = trajectory_signature[i];

    if (want.dtype != got.dtype || !want.shape.IsCompatibleWith(got.shape)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unable to create item in table '%s' since the provided trajectory "
          "is inconsistent with the table signature. The table expects column "
          "%d to be a %s %s tensor but got a %s %s tensor."
          "\n\nThe table signature is:\n\t%s"
          "\n\nThe provided trajectory signature is:\n\t%s.\n",
          table, i, tensorflow::DataTypeString(want.dtype),
          want.shape.DebugString(), tensorflow::DataTypeString(got.dtype),
          got.shape.DebugString(),
          internal::DtypesShapesString(table_signature),
          internal::DtypesShapesString(trajectory_signature)));
    }
  }

  return absl::OkStatus();
}

TrajectoryWriter::TrajectoryWriter(
    std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
    const Options& options)
    : stub_(std::move(stub)),
      options_(options),
      key_generator_(absl::make_unique<internal::UniformKeyGenerator>()),
      episode_id_(key_generator_->Generate()),
      episode_step_(0),
      closed_(false),
      stream_worker_(
          internal::StartThread("TrajectoryWriter_StreamWorker", [this] {
            absl::Duration retry_backoff = absl::Milliseconds(1);
            while (true) {
              absl::Time start_time = absl::Now();
              absl::Status status = RunStreamWorker();

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

              // If we lose the connection then we'll never receive the item
              // confirmations so we add them back to the front of the queue.
              // Note that the internal ordering of the items added to the front
              // of the queue is undefined and thus may differ from how they
              // were originally transmitted.
              for (auto& [_, item_and_refs] : in_flight_items_) {
                write_queue_.push_front(std::move(item_and_refs));
              }
              in_flight_items_.clear();

              if (absl::Now() - start_time < absl::Seconds(2)) {
                retry_backoff = std::min(absl::Seconds(1), 2 * retry_backoff);
              } else {
                retry_backoff = absl::Milliseconds(1);
              }
              auto trigger = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                return closed_;
              };
              mu_.AwaitWithTimeout(absl::Condition(&trigger), retry_backoff);
            }
          })) {
  REVERB_CHECK_OK(options.Validate());
}

TrajectoryWriter::~TrajectoryWriter() {
  {
    absl::MutexLock lock(&mu_);
    if (closed_) return;

    absl::Status status = FlushLocked(/*ignore_last_num_items=*/0,
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
  return AppendInternal(std::move(data), /*increment_episode_step=*/true, refs);
}

absl::Status TrajectoryWriter::AppendPartial(
    std::vector<absl::optional<tensorflow::Tensor>> data,
    std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs) {
  return AppendInternal(std::move(data), /*increment_episode_step=*/false,
                        refs);
}

absl::Status TrajectoryWriter::AppendInternal(
    std::vector<absl::optional<tensorflow::Tensor>> data,
    bool increment_episode_step,
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
      const tensorflow::Tensor& tensor = data[i].value();
      // If the new column has been configured with `ConfigureChunker` then we
      // use the overrided options. If not then we use the default in
      // `options_.chunker_options`.
      const std::shared_ptr<ChunkerOptions>& chunker_options =
          options_override_.contains(i) ? options_override_[i]
                                        : options_.chunker_options;
      chunkers_[i] = std::make_shared<Chunker>(
          internal::TensorSpec{std::to_string(i), tensor.dtype(),
                               tensor.shape()},
          chunker_options->Clone());
    }
  }

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
    refs->push_back(std::move(ref));
  }

  absl::MutexLock lock(&mu_);

  // Sanity check that `Append`, `AppendPartial` or `EndEpisode` wasn't called
  // concurrently.
  REVERB_CHECK_EQ(episode_info.episode_id, episode_id_);
  REVERB_CHECK_EQ(episode_info.step, episode_step_);

  if (increment_episode_step) {
    episode_step_++;
  }

  // Wake up stream worker in case it was blocked on items referencing
  // incomplete chunks
  data_cv_.Signal();

  return absl::OkStatus();
}

absl::Status TrajectoryWriter::CreateItem(
    absl::string_view table, double priority,
    absl::Span<const TrajectoryColumn> trajectory) {
  if (trajectory.empty() ||
      std::all_of(trajectory.begin(), trajectory.end(),
                  [](const TrajectoryColumn& col) { return col.empty(); })) {
    return absl::InvalidArgumentError("trajectory must not be empty.");
  }

  {
    absl::MutexLock lock(&mu_);
    REVERB_RETURN_IF_ERROR(unrecoverable_status_);
  }

  auto item_and_refs = std::make_unique<ItemAndRefs>();

  // Lock all the references to ensure that the underlying data is not
  // deallocated before the worker has successfully written the item (and data)
  // to the gRPC stream.
  for (int col_idx = 0; col_idx < trajectory.size(); ++col_idx) {
    if (absl::Status status = trajectory[col_idx].Validate(); !status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Error in column ", col_idx, ": ", status.message()));
    }
    if (!trajectory[col_idx].LockReferences(&item_and_refs->refs)) {
      return absl::InternalError("CellRef unexpectedly expired in CreateItem.");
    }
  }

  item_and_refs->item.set_key(key_generator_->Generate());
  item_and_refs->item.set_table(table.data(), table.size());
  item_and_refs->item.set_priority(priority);

  for (const TrajectoryColumn& column : trajectory) {
    column.ToProto(
        item_and_refs->item.mutable_flat_trajectory()->add_columns());
  }

  REVERB_RETURN_IF_ERROR(item_and_refs->Validate(options_));

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

void TrajectoryWriter::OnReadDone(bool ok) {
  absl::MutexLock lock(&mu_);
  data_cv_.Signal();
  if (!ok) {
    stream_ok_ = false;
    return;
  }
  for (uint64_t key : response_.keys()) {
    in_flight_items_.erase(key);
  }
  StartRead(&response_);
}

void TrajectoryWriter::OnWriteDone(bool ok) {
  absl::MutexLock lock(&mu_);
  if (ok) {
    write_inflight_ = false;
  } else {
    stream_ok_ = false;
  }
}

void TrajectoryWriter::OnDone(const ::grpc::Status& s) {
  absl::MutexLock lock(&mu_);
  stream_ok_ = false;
  stream_done_ = true;
  stream_status_ = FromGrpcStatus(s);
}

absl::Status TrajectoryWriter::Finish() {
  absl::MutexLock lock(&mu_);
  // Release a hold from SetContextAndCreateStream.
  RemoveHold();
  auto trigger = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return stream_done_;
  };
  mu_.Await(absl::Condition(&trigger));
  return stream_status_;
}

absl::Status TrajectoryWriter::SetContextAndCreateStream() {
  absl::MutexLock lock(&mu_);
  REVERB_RETURN_IF_ERROR(unrecoverable_status_);
  if (closed_) {
    return absl::CancelledError("TrajectoryWriter::Close has been called.");
  }
  context_ = absl::make_unique<grpc::ClientContext>();
  context_->set_wait_for_ready(false);
  stub_->async()->InsertStream(context_.get(), this);
  stream_ok_ = true;
  stream_done_ = false;
  // Use a hold since some StartWrites are invoked indirectly rather than
  // directly from the reactor itself.
  AddHold();
  StartRead(&response_);
  StartCall();
  return absl::OkStatus();
}

TrajectoryWriter::ItemAndRefs* TrajectoryWriter::GetNextPendingItem() {
  absl::MutexLock lock(&mu_);
  auto trigger = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return !write_queue_.empty() || closed_ || !stream_ok_;
  };
  mu_.Await(absl::Condition(&trigger));

  if (closed_ || !stream_ok_) return nullptr;
  return write_queue_.front().get();
}

bool TrajectoryWriter::SendItem(
    const internal::flat_hash_set<uint64_t>& keep_keys,
    const PrioritizedItem& item, ArenaOwnedRequest* request) {
  request->r.mutable_item()->unsafe_arena_set_allocated_item(
      const_cast<PrioritizedItem*>(&item));
  request->r.mutable_item()->set_send_confirmation(true);
  for (uint64_t keep_key : keep_keys) {
    request->r.mutable_item()->add_keep_chunk_keys(keep_key);
  }
  return Write(request);
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

    for (const std::shared_ptr<CellRef>& ref : (*it)->refs) {
      if (streamed_chunk_keys.contains(ref->chunk_key())) {
        keys.insert(ref->chunk_key());
      }
    }
  }

  return keys;
}

absl::Status TrajectoryWriter::RunStreamWorker() {
  REVERB_RETURN_IF_ERROR(SetContextAndCreateStream());
  internal::flat_hash_set<uint64_t> streamed_chunk_keys;
  ArenaOwnedRequest request;
  while (true) {
    ItemAndRefs* item_and_refs = GetNextPendingItem();
    if (item_and_refs == nullptr) {
      return Finish();
    }

    // Send referenced chunks which haven't already been sent. This call also
    // inserts the new chunk keys into `streamed_chunk_keys`.
    if (!SendNotAlreadySentChunks(&streamed_chunk_keys, item_and_refs->refs,
                                  &request)) {
      return Finish();
    }

    {
      absl::WriterMutexLock lock(&mu_);
      // Check whether all chunks referenced by the item have been written to
      // the stream. If not, then at least one chunk is incomplete and the
      // worker will wait for the chunk state to change and then retry.
      if (!ContainsAll(streamed_chunk_keys, item_and_refs->refs)) {
        // Do a final check that the chunks didn't change since the lock was
        // last held. If the item still references incomplete chunks then we
        // sleep until the chunks changed. If all the chunks are now completed
        // then we move straight to the top of the loop.
        if (!AllReady(item_and_refs->refs)) {
          data_cv_.Wait(&mu_);
        }
        continue;
      }

      // Item is about to be written - move from write_queue_ to
      // in_flight_items_.
      in_flight_items_[item_and_refs->item.key()] =
          std::move(write_queue_.front());
      write_queue_.pop_front();

      // Remove keys of expired chunks from streamed_chunk_keys to avoid OOM
      // issues caused by the otherwise indefinitely growing hash set.
      streamed_chunk_keys = GetKeepKeys(streamed_chunk_keys);
    }

    for (auto& [_, chunker] : chunkers_) {
      REVERB_RETURN_IF_ERROR(
          chunker->OnItemFinalized(item_and_refs->item, item_and_refs->refs));
    }

    // All chunks have been written to the stream so the item can now be
    // written.
    if (!SendItem(streamed_chunk_keys, item_and_refs->item, &request)) {
      return Finish();
    }
  }
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

    for (const std::shared_ptr<CellRef>& ref : item->refs) {
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
  auto cond = [ignore_last_num_items, this]()
                  ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) -> bool {
    if (!unrecoverable_status_.ok()) {
      return true;
    }
    return write_queue_.size() + in_flight_items_.size() <=
           ignore_last_num_items;
  };

  if (!mu_.AwaitWithTimeout(absl::Condition(&cond), timeout)) {
    return absl::DeadlineExceededError(
        absl::StrCat("Timeout exceeded with ", write_queue_.size(),
                     " items waiting to be written and ",
                     in_flight_items_.size(), " items awaiting confirmation."));
  }

  return unrecoverable_status_;
}

absl::Status TrajectoryWriter::EndEpisode(bool clear_buffers,
                                          absl::Duration timeout) {
  absl::MutexLock lock(&mu_);
  REVERB_RETURN_IF_ERROR(unrecoverable_status_);

  REVERB_RETURN_IF_ERROR(FlushLocked(0, timeout));

  for (auto& [_, chunker] : chunkers_) {
    if (clear_buffers) {
      chunker->Reset();
    } else {
      // This call should NEVER fail but if it does then we will not be able to
      // recover from it.
      unrecoverable_status_ = chunker->Flush();
      REVERB_RETURN_IF_ERROR(unrecoverable_status_);
    }
  }

  episode_id_ = key_generator_->Generate();
  episode_step_ = 0;
  return absl::OkStatus();
}

absl::Status TrajectoryWriter::ConfigureChunker(
    int column, const std::shared_ptr<ChunkerOptions>& options) {
  REVERB_RETURN_IF_ERROR(ValidateChunkerOptions(options.get()));

  if (auto it = chunkers_.find(column); it != chunkers_.end()) {
    return it->second->ApplyConfig(options->Clone());
  }

  options_override_[column] = options->Clone();
  return absl::OkStatus();
}

TrajectoryColumn::TrajectoryColumn(std::vector<std::weak_ptr<CellRef>> refs,
                                   bool squeeze)
    : refs_(std::move(refs)), squeeze_(squeeze) {}

void TrajectoryColumn::ToProto(FlatTrajectory::Column* proto) const {
  // Note that MergeAdjacent can safely assume that all weak_ptrs are alive
  // since the corresponding shared_ptrs exists in item_and_refs.
  for (FlatTrajectory::ChunkSlice& slice : MergeAdjacent(refs_)) {
    *proto->add_chunk_slices() = std::move(slice);
  }
  proto->set_squeeze(squeeze_);
}

absl::Status TrajectoryColumn::Validate() const {
  std::vector<std::shared_ptr<CellRef>> locked_refs;
  if (!LockReferences(&locked_refs)) {
    return absl::InvalidArgumentError("Column contains expired CellRef.");
  }

  if (squeeze_ && locked_refs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("TrajectoryColumn must contain exactly one row when "
                     "squeeze is set but got ",
                     locked_refs.size(), "."));
  }

  // Check that the column only contains compatible data references.
  const internal::TensorSpec& col_spec =
      locked_refs[0]->chunker().lock()->spec();
  for (int i = 1; i < locked_refs.size(); ++i) {
    const internal::TensorSpec& spec = locked_refs[i]->chunker().lock()->spec();
    if (spec.dtype != col_spec.dtype) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Column references tensors with different dtypes: ",
          tensorflow::DataTypeString(col_spec.dtype), " (index 0) != ",
          tensorflow::DataTypeString(spec.dtype), " (index ", i, ")."));
    }
    if (!spec.shape.IsCompatibleWith(col_spec.shape)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Column references tensors with incompatible shapes: ",
          col_spec.shape.DebugString(), " (index 0) not compatible with ",
          spec.shape.DebugString(), " (index ", i, ")."));
    }
  }

  return absl::OkStatus();
}

bool TrajectoryColumn::LockReferences(
    std::vector<std::shared_ptr<CellRef>>* locked_refs) const {
  for (const std::weak_ptr<CellRef>& ref : refs_) {
    locked_refs->push_back(ref.lock());
    if (!locked_refs->back()) return false;
  }
  return true;
}

}  // namespace reverb
}  // namespace deepmind
