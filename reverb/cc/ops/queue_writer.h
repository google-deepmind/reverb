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

#ifndef REVERB_CC_OPS_QUEUE_WRITER_H_
#define REVERB_CC_OPS_QUEUE_WRITER_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/trajectory_writer.h"
#include "tensorflow/core/framework/tensor.h"

// TODO(sabela): Add more documentation.

namespace deepmind {
namespace reverb {

// This class is not thread-safe.
class QueueWriter : public ColumnWriter {
 public:
  // The constructor takes the size of the reference buffers maintained by the
  // chunker, and a borrowed pointer to the queue where the trajectories
  // will be written.
  QueueWriter(int num_keep_alive_refs,
              std::deque<std::vector<tensorflow::Tensor>>* write_queue);

  // See `ColumnWriter::Append`.
  absl::Status Append(
      std::vector<absl::optional<tensorflow::Tensor>> data,
      std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs);

  // See `ColumnWriter::AppendPartial`.
  absl::Status AppendPartial(
      std::vector<absl::optional<tensorflow::Tensor>> data,
      std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs);

  // See `ColumnWriter::CreateItem`. `unused_table` and `unused_priority` are
  // ignored.
  absl::Status CreateItem(absl::string_view unused_table,
                          double unused_priority,
                          absl::Span<const TrajectoryColumn> trajectory);

  // See `ColumnWriter::Flush`. In this writer, there is nothing to send so this
  // is a no-op.
  absl::Status Flush(
      int unused_ignore_last_num_items = 0,
      absl::Duration unused_timeout = absl::InfiniteDuration()) override;

  // See `ColumnWriter::EndEpisode`. `unused_timeout` is ignored.
  absl::Status EndEpisode(
      bool clear_buffers,
      absl::Duration unused_timeout = absl::InfiniteDuration()) override;



  // Number of `Append` calls since last `EndEpisode` call. Note that
  // `AppendPartial` calls does not increment this counter.
  int episode_steps() const;

 private:
  // See `Append` and `AppendPartial`.
  absl::Status AppendInternal(
      std::vector<absl::optional<tensorflow::Tensor>> data,
      bool increment_episode_step,
      std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs);

  // The size of the reference buffers maintained by column chunkers.
  int num_keep_alive_refs_;

  // Mapping from column index to Chunker. Shared pointers are used as the
  // `CellRef`s created by the chunker will own a weak_ptr created using
  // `weak_from_this()` on the Chunker.
  internal::flat_hash_map<int, std::shared_ptr<Chunker>> chunkers_;

  // Trajectories that are ready to be consumed by the caller. The pointer is
  // owned by the caller.
  std::deque<std::vector<tensorflow::Tensor>>* write_queue_;

  // ID of the active episode.
  uint64_t episode_id_;

  // Step within the episode.
  int episode_step_;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_OPS_QUEUE_WRITER_H_
