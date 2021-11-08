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

#ifndef REVERB_CC_STRUCTURED_WRITER_H_
#define REVERB_CC_STRUCTURED_WRITER_H_

#include <deque>
#include <vector>

#include "absl/status/status.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/patterns.pb.h"
#include "reverb/cc/trajectory_writer.h"

namespace deepmind::reverb {

class StructuredWriter {
 public:
  StructuredWriter(std::unique_ptr<ColumnWriter> writer,
                   std::vector<StructuredWriterConfig> configs);

  // Calls `Append` on wrapped `ColumnWriter` and inserts trajectories of all
  // configs with fulfilled conditions.
  absl::Status Append(std::vector<absl::optional<tensorflow::Tensor>> data);

  // Calls `AppendPartial` on wrapped `ColumnWriter` and inserts trajectories of
  // all configs with fulfilled conditions.
  //
  // Note that patterns are never applied more than once at the same step. The
  // behaviour of calling  `Append` once with the full data is thus identical
  // to calling `AppendPartial` multiple times with partial data.
  absl::Status AppendPartial(
      std::vector<absl::optional<tensorflow::Tensor>> data);

  // Calls `EndEpisode` on wrapped `ColumnWriter`.
  absl::Status EndEpisode(bool clear_buffers,
                          absl::Duration timeout = absl::InfiniteDuration());

  // Calls `Flush` on wrapped `ColumnWriter`.
  absl::Status Flush(int ignore_last_num_items = 0,
                     absl::Duration timeout = absl::InfiniteDuration());

 private:
  // Forwards `data` to wrapped `ColumnWriter` then calls `ApplyConfig`.
  absl::Status AppendInternal(
      std::vector<absl::optional<tensorflow::Tensor>> data, bool finalize_step);

  // For each element of `configs_and_states_`, checks if ALL conditions are met
  // and if so, applies the config and inserts the item into the target table.
  absl::Status ApplyConfigs(bool is_end_of_episode);

  struct ConfigState {
    // The wrapped proto config.
    const StructuredWriterConfig config;

    // The number of since the config was last used to produce a trajectory.
    int steps_since_applied = 0;

    // The episode and step index on which this config was most recently used
    // to produce a trajectory.
    CellRef::EpisodeInfo last_applied = {0, -1};

    // The last step index that this config was last checked. This is used to
    // decide whether `steps_since_applied` should be incremented or not.
    CellRef::EpisodeInfo last_checked = {0, -1};
  };

  // True if `AppendPartial` has been called on the active step.
  bool step_is_open_ = false;

  // The actual writer which data will be forwarded to and items will be
  // inserted using.
  std::unique_ptr<ColumnWriter> writer_;

  // The age of the oldest element references in each source column references
  // by any config. That is, number of CellRef to keep alive for each column.
  // This is used to automatically clean up elements from `columns_` that are
  // guaranteed to never be used again.
  std::vector<int> max_column_history_;

  // Static configurations and details about when it was most recently applied.
  std::vector<ConfigState> configs_and_states_;

  // Each column has a buffer of the last `max_history_length_` steps.
  std::vector<std::deque<std::shared_ptr<CellRef>>> columns_;
};

// Validates the content of `config` and returns `InvalidArgumentError` if
// invalid.
absl::Status ValidateStructuredWriterConfig(
    const StructuredWriterConfig& config);

}  // namespace deepmind::reverb

#endif  // REVERB_CC_STRUCTURED_WRITER_H_
