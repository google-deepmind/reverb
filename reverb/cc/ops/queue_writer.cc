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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/support/tf_util.h"
#include "reverb/cc/support/trajectory_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"

namespace deepmind {
namespace reverb {

QueueWriter::QueueWriter(
    int num_keep_alive_refs,
    std::deque<std::vector<tensorflow::Tensor>>* write_queue)
    : num_keep_alive_refs_(num_keep_alive_refs),
      write_queue_(write_queue),
      episode_id_(0),
      episode_step_(0) {}

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
      chunkers_[i] = std::make_shared<Chunker>(
          internal::TensorSpec{std::to_string(i), tensor.dtype(),
                               tensor.shape()},
          std::make_shared<NeverCompressChunkerOptions>(num_keep_alive_refs_));
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

  std::vector<tensorflow::Tensor> out(trajectory.size());

  // Lock all the references to ensure that the underlying data is not
  // deallocated before the worker has successfully copied the item (and data)
  // to the items queue.
  for (int col_idx = 0; col_idx < trajectory.size(); ++col_idx) {
    auto& column = trajectory[col_idx];

    if (absl::Status status = column.Validate(); !status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Error in column ", col_idx, ": ", status.message()));
    }

    std::vector<std::shared_ptr<CellRef>> refs;
    if (!column.LockReferences(&refs)) {
      return absl::InternalError("CellRef unexpectedly expired in CreateItem.");
    }

    if (column.squeezed()) {
      REVERB_CHECK_EQ(refs.size(), 1);
      REVERB_RETURN_IF_ERROR(refs[0]->GetData(&out[col_idx]));
      out[col_idx] = out[col_idx].SubSlice(0);
    } else {
      std::vector<tensorflow::Tensor> column_tensors(refs.size());
      for (int ref_idx = 0; ref_idx < refs.size(); ref_idx++) {
        REVERB_RETURN_IF_ERROR(
            refs[ref_idx]->GetData(&column_tensors[ref_idx]));
      }
      REVERB_RETURN_IF_ERROR(FromTensorflowStatus(
          tensorflow::tensor::Concat(column_tensors, &out[col_idx])));
    }

    if (out[col_idx].IsAligned()) {
      out[col_idx] = tensorflow::tensor::DeepCopy(out[col_idx]);
    }
  }

  write_queue_->push_back(std::move(out));

  return absl::OkStatus();
}

absl::Status QueueWriter::EndEpisode(
    bool clear_buffers, absl::Duration unused_timeout) {
  if (clear_buffers){
    for (auto& [_, chunker] : chunkers_) {
      chunker->Reset();
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

}  // namespace reverb
}  // namespace deepmind
