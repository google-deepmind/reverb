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

#ifndef REVERB_CC_CHUNKER_H_
#define REVERB_CC_CHUNKER_H_

#include <deque>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/signature.h"
#include "tensorflow/core/framework/tensor.h"

namespace deepmind {
namespace reverb {

// TODO(b/178096736): Write high level API documentation with examples.

class CellRef;
class Chunker;

class CellRef {
 public:
  struct EpisodeInfo {
    uint64_t episode_id;
    int32_t step;
  };

  CellRef(std::weak_ptr<Chunker> chunker, uint64_t chunk_key, int offset,
          EpisodeInfo episode_info);

  // Key of the parent chunk.
  uint64_t chunk_key() const;

  // Offset within the parent chunk.
  int offset() const;

  // ID of the episode which the referenced data originated from.
  uint64_t episode_id() const;

  // Step (zero-indexed) within the episode that the data was generated at.
  int episode_step() const;

  // True if SetChunk has been called.
  bool IsReady() const ABSL_LOCKS_EXCLUDED(mu_);

  // Gets chunker if set. If not yet set then nullptr is returned.
  std::shared_ptr<const ChunkData> GetChunk() const ABSL_LOCKS_EXCLUDED(mu_);

  // Weak pointer to the parent Chunker.
  std::weak_ptr<Chunker> chunker() const;

  // Gets a copy of the referenced data. If the chunk has been finalized then
  // it is unpacked and the referenced row copied into `out`. If the chunk is
  // not yet finalized then the data is copied from the buffer of the parent
  // `Chunker`.
  absl::Status GetData(tensorflow::Tensor* out) const;

 private:
  friend Chunker;

  // Called by chunker the referenced data is flushed into a ChunkData.
  void SetChunk(std::shared_ptr<const ChunkData> chunk)
      ABSL_LOCKS_EXCLUDED(mu_);

 private:
  // Chunker which created the `CellRef` and will eventually create the chunk
  // and call `SetChunk`.
  std::weak_ptr<Chunker> chunker_;

  // Key of the parent chunk.
  uint64_t chunk_key_;

  // Offset of element within the parent chunk.
  int offset_;

  // The episode step that the referenced data was generated at.
  EpisodeInfo episode_info_;

  mutable absl::Mutex mu_;

  // Parent chunk which eventually be set by parent `Chunker`.
  absl::optional<std::shared_ptr<const ChunkData>> chunk_ ABSL_GUARDED_BY(mu_);
};

// Checks that `max_chunk_length` and `num_keep_alive_refs` is a valid `Chunker`
// configuration and returns `InvalidArgumentError` if it isn't.
absl::Status ValidateChunkerOptions(int max_chunk_length,
                                    int num_keep_alive_refs);

class Chunker : public std::enable_shared_from_this<Chunker> {
 public:
  Chunker(internal::TensorSpec spec, int max_chunk_length,
          int num_keep_alive_refs);

  // Validates `tensor` against `spec_` and `episode_info` against previous
  // calls, appends it to the active chunk and returns a reference to the new
  // row. If the active chunk now has `max_chunk_length` rows then it is
  // finalized and its `CellRef`s notified (including `ref`).
  absl::Status Append(tensorflow::Tensor tensor,
                      CellRef::EpisodeInfo episode_info,
                      std::weak_ptr<CellRef>* ref) ABSL_LOCKS_EXCLUDED(mu_);

  // Creates a chunk from the data in the buffer and calls `SetChunk` on its
  // `CellRef`s.
  absl::Status Flush() ABSL_LOCKS_EXCLUDED(mu_);

  // Clears buffers of both references and data not yet committed to a Chunk.
  void Reset();

  // Keys of the FINALIZED chunks referenced by `CellRef`s in `active_refs_`.
  std::vector<uint64_t> GetKeepKeys() const ABSL_LOCKS_EXCLUDED(mu_);

  // Spec which appended tensors need to be compatible with.
  const internal::TensorSpec& spec() const;

  // Modify options on Chunker with an empty buffer (i.e newly created or
  // `Flush` just called.). Returns `InvalidArgumentError` if
  // `max_chunk_length > num_keep_alive_refs`  or if either is <= 0.
  absl::Status ApplyConfig(int max_chunk_length, int num_keep_alive_refs)
      ABSL_LOCKS_EXCLUDED(mu_);

 private:
  friend CellRef;

  // Get the data for referenced by `ref`. If the data has been finalized into
  // a ChunkData then the chunk is unpacked and the row extracted. If the
  // chunk has not been finalized the data is copied from `buffer_`.
  absl::Status CopyDataForCell(const CellRef* ref,
                               tensorflow::Tensor* out) const;

 private:
  absl::Status FlushLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Spec which all data in `Append` must follow.
  internal::TensorSpec spec_;

  // Once the buffer reaches this size then `Flush` is automatically called.
  int max_chunk_length_;

  // Size of the buffer holding `CellRef` of most recent `Append` calls. When
  // a `CellRef` is removed from the buffer it can no longer be referenced by
  // new trajectories.
  int num_keep_alive_refs_;

  mutable absl::Mutex mu_;

  // Data waiting for the next chunk to be constructed.
  std::vector<tensorflow::Tensor> buffer_ ABSL_GUARDED_BY(mu_);

  // Offset within the chunk of the next appended item.
  int offset_ ABSL_GUARDED_BY(mu_);

  // Key of the chunk that will be constructed from `buffer_`.
  uint64_t next_chunk_key_ ABSL_GUARDED_BY(mu_);

  // Circular buffer of `CellRef`s that can be referenced in by new items.
  // When the size exceeds `num_keep_alive_refs_` then the oldest item is
  // removed.
  std::deque<std::shared_ptr<CellRef>> active_refs_ ABSL_GUARDED_BY(mu_);
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_CHUNKER_H_
