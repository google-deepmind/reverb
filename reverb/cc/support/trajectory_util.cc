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

#include "reverb/cc/support/trajectory_util.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "absl/types/span.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/tensor_compression.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace deepmind {
namespace reverb {
namespace internal {

std::vector<uint64_t> GetChunkKeys(const FlatTrajectory& trajectory) {
  std::vector<uint64_t> keys;
  internal::flat_hash_set<uint64_t> seen;
  for (const auto& col : trajectory.columns()) {
    for (const auto& slice : col.chunk_slices()) {
      if (seen.insert(slice.chunk_key()).second) {
        keys.push_back(slice.chunk_key());
      }
    }
  }
  return keys;
}

FlatTrajectory FlatTimestepTrajectory(absl::Span<const uint64_t> chunk_keys,
                                      absl::Span<const int> chunk_lengths,
                                      int num_columns, int offset, int length) {
  REVERB_CHECK_EQ(chunk_keys.size(), chunk_lengths.size());

  FlatTrajectory proto;
  while (proto.columns_size() < num_columns) {
    auto* col = proto.add_columns();
    int col_offset = offset;
    int col_remaining = length;

    for (int i = 0; i < chunk_keys.size(); i++) {
      REVERB_CHECK_GT(col_remaining, 0);
      auto* slice = col->add_chunk_slices();
      slice->set_chunk_key(chunk_keys[i]);
      slice->set_offset(col_offset);
      slice->set_length(
          std::min<int32_t>(chunk_lengths[i] - col_offset, col_remaining));
      slice->set_index(proto.columns_size() - 1);

      col_offset = 0;
      col_remaining -= slice->length();
    }

    REVERB_CHECK_EQ(col_remaining, 0);
  }

  return proto;
}

FlatTrajectory FlatTimestepTrajectory(
    absl::Span<const std::shared_ptr<ChunkStore::Chunk>> chunks, int offset,
    int length) {
  std::vector<uint64_t> chunk_keys(chunks.size());
  std::vector<int> chunk_lengths(chunks.size());
  for (int i = 0; i < chunks.size(); i++) {
    chunk_keys[i] = chunks[i]->key();
    chunk_lengths[i] = chunks[i]->num_rows();
  }
  return FlatTimestepTrajectory(chunk_keys, chunk_lengths,
                                chunks.front()->num_columns(), offset, length);
}

int ColumnLength(const FlatTrajectory& trajectory, int column) {
  REVERB_CHECK_LT(column, trajectory.columns_size());
  int length = 0;
  for (const auto& slice : trajectory.columns(column).chunk_slices()) {
    length += slice.length();
  }
  return length;
}

int TimestepTrajectoryLength(const FlatTrajectory& trajectory) {
  REVERB_CHECK(!trajectory.columns().empty());
  return ColumnLength(trajectory, 0);
}

bool IsTimestepTrajectory(const FlatTrajectory& trajectory) {
  if (trajectory.columns().empty()) {
    return false;
  }

  const auto& first_col = trajectory.columns(0);

  // The trajectory must not contain any gaps.
  for (int i = 1; i < first_col.chunk_slices_size(); i++) {
    if (first_col.chunk_slices(i).offset() > 0) {
      return false;
    }
  }

  for (int column_index = 0; column_index < trajectory.columns_size();
       ++column_index) {
    const auto& col = trajectory.columns(column_index);

    if (col.chunk_slices_size() != first_col.chunk_slices_size()) {
      return false;
    }

    for (int i = 0; i < col.chunk_slices_size(); ++i) {
      const auto& col_slice = col.chunk_slices(i);
      const auto& first_col_slice = first_col.chunk_slices(i);

      if (col_slice.chunk_key() != first_col_slice.chunk_key() ||
          col_slice.offset() != first_col_slice.offset() ||
          col_slice.length() != first_col_slice.length() ||
          col_slice.index() != column_index) {
        return false;
      }
    }
  }

  return true;
}

tensorflow::Status UnpackChunkColumn(const ChunkData& chunk_data, int column,
                                     tensorflow::Tensor* out) {
  if (column >= chunk_data.data().tensors_size() || column < 0) {
    return tensorflow::errors::InvalidArgument(
        "Cannot unpack column ", column, " in chunk ", chunk_data.chunk_key(),
        " which has ", chunk_data.data().tensors_size(), " columns.");
  }

  *out = DecompressTensorFromProto(chunk_data.data().tensors(column));
  if (chunk_data.delta_encoded()) {
    *out = DeltaEncode(*out, /*encode=*/false);
  }

  return tensorflow::Status::OK();
}

tensorflow::Status UnpackChunkColumnAndSlice(const ChunkData& chunk_data,
                                             int column, int offset, int length,
                                             tensorflow::Tensor* out) {
  TF_RETURN_IF_ERROR(UnpackChunkColumn(chunk_data, column, out));

  if (offset < 0 || offset + length > out->shape().dim_size(0)) {
    return tensorflow::errors::InvalidArgument(
        "Cannot slice (", offset, ", ", offset + length,
        ") out of tensor with shape ", out->shape().DebugString(), ".");
  }

  *out = out->Slice(offset, offset + length);
  if (!out->IsAligned()) {
    *out = tensorflow::tensor::DeepCopy(*out);
  }

  return tensorflow::Status::OK();
}

tensorflow::Status UnpackChunkColumnAndSlice(
    const ChunkData& chunk_data, const FlatTrajectory::ChunkSlice& slice,
    tensorflow::Tensor* out) {
  return UnpackChunkColumnAndSlice(chunk_data, slice.index(), slice.offset(),
                                   slice.length(), out);
}

int TimestepTrajectoryOffset(const FlatTrajectory& trajectory) {
  return trajectory.columns(0).chunk_slices(0).offset();
}

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind
