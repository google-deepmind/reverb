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
#include <numeric>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/schema.pb.h"
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

int GetLength(const ChunkData& chunk) {
  return chunk.data().tensors(0).tensor_shape().dim(0).size();
}

}  // namespace

CellRef::CellRef(std::weak_ptr<Chunker> chunker, uint64_t chunk_key, int offset,
                 CellRef::EpisodeInfo episode_info)
    : chunker_(std::move(chunker)),
      chunk_key_(chunk_key),
      offset_(offset),
      episode_info_(episode_info),
      chunk_(nullptr) {}

uint64_t CellRef::chunk_key() const { return chunk_key_; }

int CellRef::offset() const { return offset_; }

bool CellRef::IsReady() const {
  absl::MutexLock lock(&mu_);
  return chunk_ != nullptr;
}

void CellRef::SetChunk(std::shared_ptr<ChunkDataContainer> chunk) {
  absl::MutexLock lock(&mu_);
  chunk_ = std::move(chunk);
}

std::weak_ptr<Chunker> CellRef::chunker() const { return chunker_; }

std::shared_ptr<ChunkDataContainer> CellRef::GetChunk() const {
  absl::MutexLock lock(&mu_);
  return chunk_;
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

absl::Status CellRef::GetSpec(internal::TensorSpec* spec) const {
  auto chunker_sp = chunker_.lock();
  if (!chunker_sp) {
    return absl::InternalError(
        "Chunk not finalized and parent Chunker destroyed.");
  }
  *spec = chunker_sp->spec();
  return absl::OkStatus();
}

Chunker::Chunker(internal::TensorSpec spec,
                 std::shared_ptr<ChunkerOptions> options)
    : spec_(std::move(spec)),
      options_(std::move(options)),
      key_generator_(absl::make_unique<internal::UniformKeyGenerator>()) {
  REVERB_CHECK_GE(options_->GetNumKeepAliveRefs(),
                  options_->GetMaxChunkLength());
  Reset();
}

absl::Status Chunker::Append(const tensorflow::Tensor& tensor,
                             const CellRef::EpisodeInfo& episode_info,
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

  active_refs_.push_back(
      std::make_shared<CellRef>(std::weak_ptr<Chunker>(shared_from_this()),
                                next_chunk_key_, offset_++, episode_info));

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
  if (buffer_.size() >= options_->GetMaxChunkLength()) {
    REVERB_RETURN_IF_ERROR(FlushLocked());
  }

  // Delete references which which have exceeded their max age.
  while (active_refs_.size() > options_->GetNumKeepAliveRefs()) {
    active_refs_.pop_front();
  }

  *ref = active_refs_.back();

  return absl::OkStatus();
}

std::vector<uint64_t> Chunker::GetKeepKeys() const {
  absl::MutexLock lock(&mu_);
  std::vector<uint64_t> keys;
  for (const std::shared_ptr<CellRef>& ref : active_refs_) {
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

  auto chunk = absl::make_unique<ChunkData>();
  chunk->set_chunk_key(next_chunk_key_);

  tensorflow::Tensor batched;
  REVERB_RETURN_IF_ERROR(
      FromTensorflowStatus(tensorflow::tensor::Concat(buffer_, &batched)));

  if (options_->GetDeltaEncode()) {
    batched = DeltaEncode(batched, /*encode=*/true);
    chunk->set_delta_encoded(true);
  }

  CompressTensorAsProto(batched, chunk->mutable_data()->add_tensors());
  chunk->set_data_tensors_len(chunk->data().tensors_size());

  // Set the sequence range of the chunk.
  for (const auto& ref : active_refs_) {
    // active_refs_ is sorted by insertion time. Iterate over the list until
    // the first cell belonging to the newly created chunk is found.
    if (ref->chunk_key() != chunk->chunk_key()) continue;

    if (!chunk->has_sequence_range()) {
      // On the first ref belonging to this chunk, set the the episode ID and
      // set the episode length to 1 (i.e. start == end). The episode length
      // will be extended if we discover more refs belonging to this chunk.
      SequenceRange* range = chunk->mutable_sequence_range();
      range->set_episode_id(ref->episode_id());
      range->set_start(ref->episode_step());
      range->set_end(ref->episode_step());
    } else {
      SequenceRange* range = chunk->mutable_sequence_range();

      // Sanity check: The ref belongs to this episode (and chunk) and the ref's
      // step counter is monotonically increasing (i.e. active_refs_ is sorted
      // by insertion time).
      REVERB_CHECK(range->episode_id() == ref->episode_id() &&
                   range->end() < ref->episode_step());

      // The chunk is sparse if not all steps are represented in the data.

      // Dense chunks have subsequent step counters:
      // ref(episode_step=0), ref(episode_step=1), ...
      // Sparse chunks have holes in their step counters:
      // ref(episode_step=0), ref(episode_step=42), ...
      // We detect sparse chunks by looking at the step increments.
      // range->end() is the episode_step of the previous ref in this chunk.
      if (ref->episode_step() != range->end() + 1) {
        range->set_sparse(true);
      }
      range->set_end(ref->episode_step());
    }
  }

  // Now the chunk has been finalized we can notify the `CellRef`s.
  auto chunk_container = std::make_shared<ChunkDataContainer>(std::move(chunk));
  for (std::shared_ptr<CellRef>& ref : active_refs_) {
    if (ref->chunk_key() == chunk_container->chunk->chunk_key()) {
      ref->SetChunk(chunk_container);
    }
  }

  buffer_.clear();
  next_chunk_key_ = key_generator_->Generate();
  offset_ = 0;

  return absl::OkStatus();
}

void Chunker::Reset() {
  absl::MutexLock lock(&mu_);
  buffer_.clear();
  buffer_.reserve(options_->GetMaxChunkLength());
  offset_ = 0;
  next_chunk_key_ = key_generator_->Generate();
  active_refs_.clear();
}

const internal::TensorSpec& Chunker::spec() const { return spec_; }

absl::Status Chunker::ApplyConfig(std::shared_ptr<ChunkerOptions> options) {
  absl::MutexLock lock(&mu_);

  if (!buffer_.empty()) {
    return absl::FailedPreconditionError(
        "Flush must be called before ApplyConfig.");
  }

  REVERB_RETURN_IF_ERROR(ValidateChunkerOptions(options.get()));
  options_ = std::move(options);

  while (active_refs_.size() > options_->GetNumKeepAliveRefs()) {
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
        internal::UnpackChunkColumn(*ref->GetChunk()->get(), 0, &column));
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

void Chunker::OnItemFinalized(const PrioritizedItem& item,
                              absl::Span<const std::shared_ptr<CellRef>> refs) {
  std::vector<std::shared_ptr<CellRef>> child_refs;
  for (const auto& ref : refs) {
    auto chunker_sp = ref->chunker().lock();
    REVERB_CHECK(chunker_sp);
    if (chunker_sp.get() == this) {
      child_refs.push_back(ref);
    }
  }
  if (!child_refs.empty()) {
    options_->OnItemFinalized(item, child_refs);
  }
}

absl::Status ValidateChunkerOptions(const ChunkerOptions* options) {
  if (options->GetMaxChunkLength() <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("max_chunk_length must be > 0 but got ",
                     options->GetMaxChunkLength(), "."));
  }
  if (options->GetNumKeepAliveRefs() <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("num_keep_alive_refs must be > 0 but got ",
                     options->GetNumKeepAliveRefs(), "."));
  }
  if (options->GetMaxChunkLength() > options->GetNumKeepAliveRefs()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "num_keep_alive_refs (", options->GetNumKeepAliveRefs(),
        ") must be >= max_chunk_length (", options->GetMaxChunkLength(), ")."));
  }
  return absl::OkStatus();
}

ConstantChunkerOptions::ConstantChunkerOptions(int max_chunk_length,
                                               int num_keep_alive_refs,
                                               bool delta_encode)
    : max_chunk_length_(max_chunk_length),
      num_keep_alive_refs_(num_keep_alive_refs),
      delta_encode_(delta_encode) {}

int ConstantChunkerOptions::GetMaxChunkLength() const {
  return max_chunk_length_;
}

int ConstantChunkerOptions::GetNumKeepAliveRefs() const {
  return num_keep_alive_refs_;
}

bool ConstantChunkerOptions::GetDeltaEncode() const { return delta_encode_; }

void ConstantChunkerOptions::OnItemFinalized(
    const PrioritizedItem& item,
    absl::Span<const std::shared_ptr<CellRef>> refs) {}

std::shared_ptr<ChunkerOptions> ConstantChunkerOptions::Clone() const {
  return std::make_shared<ConstantChunkerOptions>(max_chunk_length_,
                                                  num_keep_alive_refs_);
}

AutoTunedChunkerOptions::AutoTunedChunkerOptions(int num_keep_alive_refs,
                                                 double throughput_weight,
                                                 bool delta_encode)
    : num_keep_alive_refs_(num_keep_alive_refs),
      delta_encode_(delta_encode),
      throughput_weight_(throughput_weight),
      max_chunk_length_(1),
      prev_score_(Score{-1, -1}) {}

int AutoTunedChunkerOptions::GetMaxChunkLength() const {
  absl::MutexLock lock(&mu_);
  return max_chunk_length_;
}

int AutoTunedChunkerOptions::GetNumKeepAliveRefs() const {
  return num_keep_alive_refs_;
}

bool AutoTunedChunkerOptions::GetDeltaEncode() const { return delta_encode_; }

void AutoTunedChunkerOptions::PushItem(
    absl::Span<const std::shared_ptr<CellRef>> refs) {
  double total_bytes = 0;
  double total_chunk_length = 0;

  internal::flat_hash_set<uint64_t> seen_chunks;
  for (const auto& ref : refs) {
    if (seen_chunks.insert(ref->chunk_key()).second) {
      total_bytes += ref->GetChunk()->get()->ByteSizeLong();
      total_chunk_length += GetLength(*ref->GetChunk()->get());
    }
  }

  Statistic summary;
  summary.average_chunk_length = total_chunk_length / seen_chunks.size();
  summary.bytes_per_step = total_bytes / refs.size();
  items_.push_back(std::move(summary));

  if (items_.size() > kNumItemsToScore) {
    items_.pop_front();
  }
}

void AutoTunedChunkerOptions::OnItemFinalized(
    const PrioritizedItem& item,
    absl::Span<const std::shared_ptr<CellRef>> refs) {
  REVERB_CHECK(!refs.empty());

  absl::MutexLock lock(&mu_);

  // Push items and chunks to history buffers.
  PushItem(refs);
  PushChunks(refs);

  // If there isn't enough examples yet then don't make any changes.
  if (items_.size() < kNumItemsToScore || chunks_.size() < kNumChunksToScore) {
    return;
  }

  auto new_score = ReduceAndClearBuffers();
  items_.clear();
  chunks_.clear();

  // If this is the first time the score has been recorded then we increase the
  // `max_chunk_length_` so there is something to compare to the next time the
  // score is calculated.
  if (prev_score_.average_chunk_length == -1) {
    prev_score_ = new_score;
    max_chunk_length_ = std::min(max_chunk_length_ + kPosMaxChunkLengthDiff,
                                 num_keep_alive_refs_);
    return;
  }

  // If the needle hasn't moved enough then remove the oldest item from the
  // buffer and wait for the next item.
  if (std::abs(new_score.average_chunk_length - max_chunk_length_) >
      kMaxChunkLengthError) {
    return;
  }

  bool cost_reduced = new_score.cost < prev_score_.cost;
  bool length_grew =
      new_score.average_chunk_length > prev_score_.average_chunk_length;

  int diff = cost_reduced == length_grew ? kPosMaxChunkLengthDiff
                                         : kNegMaxChunkLengthDiff;
  int new_max_chunk_length =
      std::min(std::max(max_chunk_length_ + diff, 1), num_keep_alive_refs_);

  if (new_max_chunk_length != max_chunk_length_) {
    prev_score_ = new_score;
    max_chunk_length_ = new_max_chunk_length;
  }
}

AutoTunedChunkerOptions::Score
AutoTunedChunkerOptions::ReduceAndClearBuffers() {
  // We can't use REVERB_CHECK_EQ here since takes arguments by reference and
  // you can't take the address of a static member which doesn't have an
  // out-of-class definition (https://www.stroustrup.com/bs_faq2.html#in-class).
  REVERB_CHECK(items_.size() == kNumItemsToScore);
  REVERB_CHECK(chunks_.size() == kNumChunksToScore);

  double avg_chunk_length_sum = 0;

  double avg_bytes_per_item_step = 0;
  for (const auto& summary : items_) {
    avg_bytes_per_item_step += summary.bytes_per_step / items_.size();
    avg_chunk_length_sum += summary.average_chunk_length;
  }

  double avg_bytes_per_chunk_step = 0;
  for (const auto& summary : chunks_) {
    avg_bytes_per_chunk_step += summary.bytes_per_step / chunks_.size();
    avg_chunk_length_sum += summary.average_chunk_length;
  }

  Score score = {
      avg_chunk_length_sum / (items_.size() + chunks_.size()),
      avg_bytes_per_chunk_step + avg_bytes_per_item_step * throughput_weight_,
  };

  items_.clear();
  chunks_.clear();

  return score;
}

void AutoTunedChunkerOptions::PushChunks(
    absl::Span<const std::shared_ptr<CellRef>> refs) {
  for (const auto& ref : refs) {
    const ChunkData& chunk = *ref->GetChunk()->get();
    if (std::all_of(chunks_.begin(), chunks_.end(), [&chunk](const auto& s) {
          return s.key != chunk.chunk_key();
        })) {
      Statistic summary;
      summary.key = chunk.chunk_key();
      summary.average_chunk_length = GetLength(chunk);
      summary.bytes_per_step =
          chunk.ByteSizeLong() / summary.average_chunk_length;
      chunks_.push_back(std::move(summary));
    }
  }

  while (chunks_.size() > 5) {
    chunks_.pop_front();
  }
}

std::shared_ptr<ChunkerOptions> AutoTunedChunkerOptions::Clone() const {
  return std::make_shared<AutoTunedChunkerOptions>(num_keep_alive_refs_,
                                                   throughput_weight_);
}

}  // namespace reverb
}  // namespace deepmind
