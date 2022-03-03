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

#include "reverb/cc/ops/queue_writer.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <vector>

#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "absl/status/statusor.h"
#include "reverb/cc/support/tf_util.h"
#include "reverb/cc/support/trajectory_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"

namespace deepmind {
namespace reverb {
namespace {

absl::StatusOr<std::vector<tensorflow::Tensor>> UnpackColumns(
    std::vector<std::vector<tensorflow::Tensor>> columns) {
  // TODO(sabela): This code is the same as the one used to unpack samples.
  // We can simplify this further by avoiding serialization in the chunker.
  std::vector<tensorflow::Tensor> data(columns.size());
  for (int i = 0; i < columns.size(); i++) {
    // If the column is made up of a single batched tensor then there will be no
    // need for concatenation so we can save ourselves a copy by simply moving
    // the one (unpacked) chunk into sequences.
    // TODO(sabela): so far we dont have any test with larger column size
    if (columns[i].size() == 1) {
      data[i] = std::move(columns[i][0]);
    } else {
      std::vector<tensorflow::Tensor> column_tensors(
          std::make_move_iterator(columns[i].begin()),
          std::make_move_iterator(columns[i].end()));
      REVERB_RETURN_IF_ERROR(FromTensorflowStatus(
          tensorflow::tensor::Concat(column_tensors, &data.at(i))));
    }
  }
  return std::move(data);
}

// Returns true if all references `refs` are ready.
bool AllReady(absl::Span<const std::shared_ptr<CellRef>> refs) {
  return absl::c_all_of(refs, [](const auto& ref) { return ref->IsReady(); });
}

}  // namespace

absl::Status QueueWriter::Options::Validate() const {
  if (chunker_options == nullptr) {
    return absl::InvalidArgumentError("chunker_options must be set.");
  }
  return ValidateChunkerOptions(chunker_options.get());
}

QueueWriter::QueueWriter(
    const Options& options,
    std::deque<std::vector<tensorflow::Tensor>>* write_queue)
    : options_(options),
      write_queue_(write_queue),
      episode_id_(0),
      episode_step_(0) {
  REVERB_CHECK_OK(options.Validate());
}

absl::Status QueueWriter::Append(
    std::vector<absl::optional<tensorflow::Tensor>> data,
    std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs) {
  return AppendInternal(std::move(data), /*increment_episode_step=*/true, refs);
}

absl::Status QueueWriter::AppendPartial(
    std::vector<absl::optional<tensorflow::Tensor>> data,
    std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs) {
  return AppendInternal(std::move(data), /*increment_episode_step=*/false,
                        refs);
}

absl::Status QueueWriter::AppendInternal(
    std::vector<absl::optional<tensorflow::Tensor>> data,
    bool increment_episode_step,
    std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs) {
  CellRef::EpisodeInfo episode_info;
  episode_info = {episode_id_, episode_step_};

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

  // Sanity check that `Append`, `AppendPartial` or `EndEpisode` wasn't called
  // concurrently.
  REVERB_CHECK_EQ(episode_info.episode_id, episode_id_);
  REVERB_CHECK_EQ(episode_info.step, episode_step_);

  if (increment_episode_step) {
    episode_step_++;
  }
  return absl::OkStatus();
}

absl::Status QueueWriter::CreateItem(
    absl::string_view unused_table, double unused_priority,
    absl::Span<const TrajectoryColumn> trajectory) {
  if (trajectory.empty() ||
      std::all_of(trajectory.begin(), trajectory.end(),
                  [](const TrajectoryColumn& col) { return col.empty(); })) {
    return absl::InvalidArgumentError("trajectory must not be empty.");
  }
  ItemAndRefs item_and_refs;
  // Lock all the references to ensure that the underlying data is not
  // deallocated before the worker has successfully copied the item (and data)
  // to the items queue.
  for (int col_idx = 0; col_idx < trajectory.size(); ++col_idx) {
    if (absl::Status status = trajectory[col_idx].Validate(); !status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Error in column ", col_idx, ": ", status.message()));
    }
    if (!trajectory[col_idx].LockReferences(&item_and_refs.refs)) {
      return absl::InternalError("CellRef unexpectedly expired in CreateItem.");
    }
  }
  for (const TrajectoryColumn& column : trajectory) {
    column.ToProto(
        item_and_refs.item.mutable_flat_trajectory()->add_columns());
  }
  if (!AllReady(item_and_refs.refs)) {
    for (const std::shared_ptr<CellRef>& ref : item_and_refs.refs) {
      if (!ref->IsReady()) {
        REVERB_RETURN_IF_ERROR(ref->chunker().lock()->Flush());
      }
    }
  }
  return CreateTrajectoryFromItem(std::move(item_and_refs));
}

absl::Status QueueWriter::EndEpisode(
    bool clear_buffers, absl::Duration unused_timeout) {
  for (auto& [_, chunker] : chunkers_) {
    if (clear_buffers) {
      chunker->Reset();
    } else {
      // This call should NEVER fail.
      REVERB_RETURN_IF_ERROR(chunker->Flush());
    }
  }

  episode_id_++;
  episode_step_ = 0;
  return absl::OkStatus();
}

int QueueWriter::episode_steps() const { return episode_step_; }


absl::Status QueueWriter::Flush(int unused_ignore_last_num_items,
                                            absl::Duration unused_timeout) {
  // Nothing to do here. Items are put in the queue as they are created.
  return absl::OkStatus();
}

absl::Status QueueWriter::CreateTrajectoryFromItem(
    ItemAndRefs item_and_refs) {
  // Convert into a Sample.
  internal::flat_hash_map<uint64_t, std::shared_ptr<ChunkDataContainer>> chunks;
  for (auto& ref : item_and_refs.refs) {
    chunks[ref->chunk_key()] = ref->GetChunk();
  }
  std::vector<std::vector<tensorflow::Tensor>> column_chunks;

  for (const auto& column : item_and_refs.item.flat_trajectory().columns()) {
    std::vector<tensorflow::Tensor> unpacked_chunks;
    unpacked_chunks.reserve(column.chunk_slices().size());
    for (const auto& slice : column.chunk_slices()) {
      unpacked_chunks.emplace_back();
      REVERB_RETURN_IF_ERROR(internal::UnpackChunkColumnAndSlice(
          *chunks[slice.chunk_key()]->get(), slice, &unpacked_chunks.back()));
    }
    column_chunks.push_back(std::move(unpacked_chunks));
  }
  std::vector<bool> squeeze_columns;
  squeeze_columns.reserve(item_and_refs.item.flat_trajectory().columns_size());
  for (const auto& col : item_and_refs.item.flat_trajectory().columns()) {
    squeeze_columns.push_back(col.squeeze());
  }

  // Convert into a Trajectory.
  REVERB_ASSIGN_OR_RETURN(std::vector<tensorflow::Tensor> sequences,
                          UnpackColumns(std::move(column_chunks)));
  // Remove batch dimension from squeezed columns.
  for (int i = 0; i < squeeze_columns.size(); i++) {
    if (!squeeze_columns[i]) continue;
    if (int batch_dim = sequences[i].shape().dim_size(0); batch_dim != 1) {
      return absl::InternalError(absl::StrCat(
          "Tried to squeeze column with batch size ", batch_dim, "."));
    }

    sequences[i] = sequences[i].SubSlice(0);
    // TODO(sabela): so far we dont have any test with misaligned data
    if (!sequences[i].IsAligned()) {
      sequences[i] = tensorflow::tensor::DeepCopy(sequences[i]);
    }
  }
  write_queue_->push_back(std::move(sequences));

  // Call on item finalized to trigger the chunk cleanup if needed.
  for (auto& [_, chunker] : chunkers_) {
    REVERB_RETURN_IF_ERROR(
        chunker->OnItemFinalized(item_and_refs.item, item_and_refs.refs));
  }
  return absl::OkStatus();
}

}  // namespace reverb
}  // namespace deepmind
