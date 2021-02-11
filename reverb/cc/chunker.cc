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

#include "reverb/cc/chunker.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/support/tf_util.h"
#include "reverb/cc/support/trajectory_util.h"
#include "reverb/cc/tensor_compression.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"

namespace deepmind {
namespace reverb {
namespace {

// TODO(b/178091431): Move this into the classes and potentially make it
// injectable so it can be overidden in tests.
uint64_t NewKey() {
  absl::BitGen gen;
  return absl::Uniform<uint64_t>(gen, 0, std::numeric_limits<uint64_t>::max());
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

absl::Status CellRef::GetData(tensorflow::Tensor* out) const {
  auto chunker_sp = chunker_.lock();
  if (!chunker_sp) {
    return absl::InternalError(
        "Chunk not finalized and parent Chunker destroyed.");
  }
  return chunker_sp->CopyDataForCell(this, out);
}

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

  REVERB_RETURN_IF_ERROR(
      ValidateChunkerOptions(max_chunk_length, num_keep_alive_refs));

  max_chunk_length_ = max_chunk_length;
  num_keep_alive_refs_ = num_keep_alive_refs;

  while (active_refs_.size() > num_keep_alive_refs) {
    active_refs_.pop_front();
  }

  return absl::OkStatus();
}

absl::Status Chunker::CopyDataForCell(const CellRef* ref,
                                      tensorflow::Tensor* out) const {
  absl::MutexLock lock(&mu_);

  // If the chunk has been finalized then we unpack it and slice out the data.
  if (ref->IsReady()) {
    tensorflow::Tensor column;
    REVERB_RETURN_IF_ERROR(
        internal::UnpackChunkColumn(*ref->GetChunk(), 0, &column));
    *out = column.SubSlice(ref->offset());
    if (!out->IsAligned()) {
      *out = tensorflow::tensor::DeepCopy(*out);
    }
    return absl::OkStatus();
  }

  // Since the chunk hasn't been finalized then the data should be in the
  // buffer. We iterate backward over the active references until we find `ref`
  // to determine which position in the buffer holds the data.
  int negative_offset = 0;
  for (auto it = active_refs_.crbegin(); it != active_refs_.crend(); it++) {
    if (it->get() == ref) break;
    negative_offset++;
  }

  int buffer_index = buffer_.size() - negative_offset - 1;
  if (buffer_index < 0) {
    return absl::InternalError(
        "Data could not be found in buffer nor in finalized chunk.");
  }

  // A batch dimension is added to the data before it is added to the buffer so
  // we strip that off before copying the content to the output tensor.
  tensorflow::TensorShape shape = buffer_[buffer_index].shape();
  shape.RemoveDim(0);
  if (!out->CopyFrom(buffer_[buffer_index], shape)) {
    return absl::InternalError("Unable to copy tensor from buffer.");
  }

  return absl::OkStatus();
}

absl::Status ValidateChunkerOptions(int max_chunk_length,
                                    int num_keep_alive_refs) {
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
