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

#ifndef REVERB_CC_SUPPORT_TRAJECTORY_UTIL_H_
#define REVERB_CC_SUPPORT_TRAJECTORY_UTIL_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/schema.pb.h"
#include "tensorflow/core/framework/tensor.h"

namespace deepmind {
namespace reverb {
namespace internal {

// Extract and dedup the keys of all referenced chunks.
std::vector<uint64_t> GetChunkKeys(const FlatTrajectory& trajectory);

// Helper for building a FlatTrajectory of timesteps where chunks contain data
// for all the columns. The trajectory starts at step `offset` within the first
// chunk and continues for `length` steps, possibly leaving out one or more
// steps at the end of the last chunk.
FlatTrajectory FlatTimestepTrajectory(
    absl::Span<const std::shared_ptr<ChunkStore::Chunk>> chunks, int offset,
    int length);

// Helper for building a FlatTrajectory of timesteps where chunks contain data
// for all the columns. The trajectory starts at step `offset` within the first
// chunk and continues for `length` steps, possibly leaving out one or more
// steps at the end of the last chunk.
FlatTrajectory FlatTimestepTrajectory(absl::Span<const uint64_t> chunk_keys,
                                      absl::Span<const int> chunk_lengths,
                                      int num_columns, int offset, int length);

// Checks if trajectory is made up by consecutive steps where the data is owned
// by chunks of timesteps rather than chunks of columns.
bool IsTimestepTrajectory(const FlatTrajectory& trajectory);

// Alias for ColumnLength(?, 0). Assumes that IsTimestepTrajectory has been
// checked by caller before.
int TimestepTrajectoryLength(const FlatTrajectory& trajectory);

// Alias for trajectory.columns(0).chunk_slices(0).offset(). Assumes that
// IsTimestepTrajectory has been called before.
int TimestepTrajectoryOffset(const FlatTrajectory& trajectory);

// Number of steps referenced by column.
int ColumnLength(const FlatTrajectory& trajectory, int column);

// Decompresses the tensor at index `column` in `chunk_data` into `out`.
absl::Status UnpackChunkColumn(const ChunkData& chunk_data, int column,
                               tensorflow::Tensor* out);

// Unpacks content of column (see `UnpackChunkColumn`) and returns an aligned
// tensor of the desired slice,
absl::Status UnpackChunkColumnAndSlice(const ChunkData& chunk_data, int column,
                                       int offset, int length,
                                       tensorflow::Tensor* out);

// Unpacks content of column (see `UnpackChunkColumn`) and returns an aligned
// tensor of the desired slice,
absl::Status UnpackChunkColumnAndSlice(const ChunkData& chunk_data,
                                       const FlatTrajectory::ChunkSlice& slice,
                                       tensorflow::Tensor* out);

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_TRAJECTORY_UTIL_H_
