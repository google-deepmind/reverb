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
#include "absl/types/span.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/key_generators.h"
#include "reverb/cc/support/signature.h"
#include "tensorflow/core/framework/tensor.h"

namespace deepmind {
namespace reverb {

// TODO(b/178096736): Write high level API documentation with examples.

class CellRef;
class Chunker;
class ChunkerOptions;

// Small wrapper around a unique_ptr to make ownership of the ChunkData proto
// explicit. Usually, the ChunkData proto is owned by the collection of
// `CellRef` instances that belong to the chunk and each instance has a
// shared_ptr pointing to the container. However, by releasing the unique_ptr
// it's possible to transfer ownership of the ChunkData proto from the
// collection of `CellRef` instances to a new owner. This is useful if we want
// to reduce memory usage by deleting ChunkData as soon as the data is streamed.
struct ChunkDataContainer {
  explicit ChunkDataContainer(std::unique_ptr<ChunkData> chunk)
      : chunk(std::move(chunk)) {}

  const ChunkData* get() const {
    REVERB_CHECK(chunk != nullptr)
        << "Chunk data was deleted. This usually happens when using "
           "StreamingTrajectoryWriter, which releases memory greedily.";
    return chunk.get();
  }

  std::unique_ptr<const ChunkData> chunk;
};

// References a single cell (i.e. a single tensor) in a data column that was
// added to Reverb. `CellRef`s are created by a `Chunker` instance when
// aggregating trajectories. A `CellRef` indexes into a chunk, which is a
// continuous subset of rows of a column. `chunk_key` references the chunk and
// `offset` identifies the offset within the chunk (starting at 0).
//
// The actual chunk referenced by `CellRef` is typically only created after the
// `CellRef` instance was created. This is because the `Chunker` waits until a
// sufficient batch size has been achieved before actually creating the chunk.
// As such, the data referenced by `CellRef` initially lives inside the
// `Chunker`. Once the chunker has created a chunk, the chunk is associated with
// the corresponding `CellRef`s via the `SetChunk` member function.
//
// Note: Chunks consume a lot of memory since they hold the actual tensor data.
// They are kept in memory until the last `CellRef` is destroyed or deletes its
// reference to the chunk.
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
  std::shared_ptr<ChunkDataContainer> GetChunk() const ABSL_LOCKS_EXCLUDED(mu_);

  // Weak pointer to the parent Chunker.
  std::weak_ptr<Chunker> chunker() const;

  // Gets a copy of the referenced data. If the chunk has been finalized then
  // it is unpacked and the referenced row copied into `out`. If the chunk is
  // not yet finalized then the data is copied from the buffer of the parent
  // `Chunker`.
  absl::Status GetData(tensorflow::Tensor* out) const;

  // Gets the internal::TensorSpec for the referenced data. This provides a
  // description of the referenced data's dtype and shape information. This
  // will raise an error if the Chunk is not finalized and the parent Chunker
  // has been destroyed.
  absl::Status GetSpec(internal::TensorSpec* spec) const;

 private:
  friend Chunker;

  // Called by chunker the referenced data is flushed into a ChunkData.
  void SetChunk(std::shared_ptr<ChunkDataContainer> chunk)
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

  // The chunk, which holds the data referenced by this `CellRef` instance.
  // nullptr until the chunk is actually created by the parent `Chunker` and
  // updated via `SetChunk`.
  std::shared_ptr<ChunkDataContainer> chunk_ ABSL_GUARDED_BY(mu_);
};

// Checks that `max_chunk_length` and `num_keep_alive_refs` is a valid `Chunker`
// configuration and returns `InvalidArgumentError` if it isn't.
absl::Status ValidateChunkerOptions(const ChunkerOptions* options);

class Chunker : public std::enable_shared_from_this<Chunker> {
 public:
  Chunker(internal::TensorSpec spec, std::shared_ptr<ChunkerOptions> options);

  // Validates `tensor` against `spec_` and `episode_info` against previous
  // calls, appends it to the active chunk and returns a reference to the new
  // row. If the active chunk now has `max_chunk_length` rows then it is
  // finalized and its `CellRef`s notified (including `ref`).
  absl::Status Append(const tensorflow::Tensor& tensor,
                      const CellRef::EpisodeInfo& episode_info,
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
  absl::Status ApplyConfig(std::shared_ptr<ChunkerOptions> options)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Called by parent `TrajectoryWriter` when an item has been finalized. If
  // any of `refs` was created by this `Chunker` then `item` and a filtered
  // (only the ones created by this `Chunker`) vector of `refs` is forwarded to
  // the `ChunkerOptions` so it can adapt.
  void OnItemFinalized(const PrioritizedItem& item,
                       absl::Span<const std::shared_ptr<CellRef>> refs);

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

  // Provides max chunk length and the number of references to keep alive.
  // Values may change over time depending on the implementation.
  std::shared_ptr<ChunkerOptions> options_;

  mutable absl::Mutex mu_;

  // Data waiting for the next chunk to be constructed.
  std::vector<tensorflow::Tensor> buffer_ ABSL_GUARDED_BY(mu_);

  // Offset within the chunk of the next appended item.
  int offset_ ABSL_GUARDED_BY(mu_);

  // Key of the chunk that will be constructed from `buffer_`.
  uint64_t next_chunk_key_ ABSL_GUARDED_BY(mu_);

  // Used to generate chunk keys.
  std::unique_ptr<internal::KeyGenerator> key_generator_;

  // Circular buffer of `CellRef`s that can be referenced in by new items.
  // When the size exceeds `num_keep_alive_refs_` then the oldest item is
  // removed.
  std::deque<std::shared_ptr<CellRef>> active_refs_ ABSL_GUARDED_BY(mu_);
};

class ChunkerOptions {
 public:
  virtual ~ChunkerOptions() = default;

  // Get current recommendation of `max_chunk_length`.
  //
  // Once the buffer reaches `max_chunk_length` items then `Flush` is
  // automatically called.
  virtual int GetMaxChunkLength() const = 0;

  // Get current recommendation of `num_keep_alive_refs`.
  //
  // `num_keep_alive_refs` is the size of the buffer holding `CellRef` of the
  // most recent `Append` calls. When a `CellRef` is removed from the buffer it
  // can no longer be referenced by new trajectories.
  virtual int GetNumKeepAliveRefs() const = 0;

  // Get current recommendation of whether delta encoding should be used.
  virtual bool GetDeltaEncode() const = 0;

  // Called by parent `Chunker` once an item is ready to be sent to the
  // server.
  //
  // Implementations can extract performance features from these calls and
  // use it for to select the responses of future `GetMaxChunkLength` and
  // `GetNumKeepAliveRefs` calls.
  //
  // `item` is the table item scheduled for insertion. The `flat_trajectory`
  //   field in particular is likely to be of interest for selecting good
  //   chunk lengths.
  // `refs` are the `CellRef` created by the parent `Chunker` and referenced
  // by
  //   `item`.
  //
  virtual void OnItemFinalized(
      const PrioritizedItem& item,
      absl::Span<const std::shared_ptr<CellRef>> refs) = 0;

  // Make a copy of this `ChunkerOptions` and state. This allows a particular
  // implementation
  // to be used as a template for all (or some) of the `Chunker`s owned by a
  // `TrajectoryWriter`.
  virtual std::shared_ptr<ChunkerOptions> Clone() const = 0;
};

// Returns a constant `max_chunk_length` and `num_keep_alive_refs`.
// `OnItemFinalized` is a noop.
class ConstantChunkerOptions : public ChunkerOptions {
 public:
  ConstantChunkerOptions(int max_chunk_length, int num_keep_alive_refs,
                         bool delta_encode = false);

  int GetMaxChunkLength() const override;

  int GetNumKeepAliveRefs() const override;

  bool GetDeltaEncode() const override;

  void OnItemFinalized(
      const PrioritizedItem& item,
      absl::Span<const std::shared_ptr<CellRef>> refs) override;

  std::shared_ptr<ChunkerOptions> Clone() const override;

 private:
  int max_chunk_length_;
  int num_keep_alive_refs_;
  bool delta_encode_;
};

// Automatically tunes the `max_chunk_length` value within the range [1,
// `num_keep_alive_refs`] by minimizing a score based on the number of bytes
// sent per step in items and chunks.
//
// The score is calculated by maintaining a buffer of most recently observed
// items and chunks. Once the buffer contains `kNumItemsToScore` items and
// `kNumChunksToScore` chunks then a cost-score is calculated as follows. After
// the score has been calculated the buffers are cleared:
//
//   bytes_per_step_in_item * throughput_weight + bytes_per_step_in_chunks
//
// Both the item and chunk `bytes_per_step_*` are averages for all elements in
// the buffer.
//
// When a buffer has been scored it is compared against the most recently score
// before that. A lower score is better so if the new score is lower then the
// `max_chunk_length` is moved in the direction of `new_average_chunk_length -
// prev_average_chunk_length`. For example, if the previous score was 100 and
// the average chunk length at the time was 5 and the new score is 50 with a
// new average chunk length of 6 then `max_chunk_length` is increased. If the
// score is 75 (average chunk length is 7) the next time it is calculated then
// `max_chunk_length` is decreased.
class AutoTunedChunkerOptions : public ChunkerOptions {
 public:
  // Wait until the buffer includes this many items and chunks before scoring
  // the performance and (potentially) updating the recommendations for
  // `GetMaxChunkLength`. If a buffers exceeds this value after a new element
  // have been pushed then the oldest element is removed.
  static const int kNumItemsToScore = 10;
  static const int kNumChunksToScore = 5;

  // Diff added to the current max chunk length when increasing or decreasing it
  // as a response to observed data. New values are clipped to ensure that the
  // updated value is within the range (1, `num_keep_alive_refs`).
  static const int kPosMaxChunkLengthDiff = 2;
  static const int kNegMaxChunkLengthDiff = -1;

  // Maximum difference between the average observed chunk length and the
  // current recommendation of `GetMaxChunkLength` required for a score to be
  // considered as valid. If the difference is larger than this value then the
  // score is ignored and the content of the buffers dropped.
  static constexpr auto kMaxChunkLengthError = 0.25;

  // TODO(b/180278134): Remove delta_encode argument once it is auto selected.
  explicit AutoTunedChunkerOptions(int num_keep_alive_ref,
                                   double throughput_weight = 1.0,
                                   bool delta_encode = false);

  // Returns the recommendation of the maximum chunk length.
  int GetMaxChunkLength() const override;

  // Returns the (constant) size of the reference buffer.
  int GetNumKeepAliveRefs() const override;

  // Returns the (constant) delta encoding setting.
  bool GetDeltaEncode() const override;

  // Calculates performance statistics for the item and the chunks it
  // reference and uses thse to (potentially) update the result of
  // `GetMaxChunkLength`.
  void OnItemFinalized(
      const PrioritizedItem& item,
      absl::Span<const std::shared_ptr<CellRef>> refs) override;

  std::shared_ptr<ChunkerOptions> Clone() const override;

 private:
  struct Score;

  // Appends a `Statistic` for every referenced chunk which isn't already part
  // of `chunks`.
  void PushChunks(absl::Span<const std::shared_ptr<CellRef>> refs)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Appends a `Statistic` of the item to `items_`.
  void PushItem(absl::Span<const std::shared_ptr<CellRef>> refs)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Calculates an overall score for the data in a FULL buffer then clears both
  // `items_` and `chunks_`.
  Score ReduceAndClearBuffers() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // The maximum number of CellRef to keep alive. This value is NOT tuned.
  int num_keep_alive_refs_;

  // Whethr delta encoding should be used. This value is NOT tuned.
  bool delta_encode_;

  // Weight to multiply the score contribution from `items_` with. A higher
  // value results in more emphasise on the amount of data sent per item (i.e
  // sample speed) and lower values results in lower memory usage on the server
  // (i.e maximize impact of compression across all of the data rather than
  // focusing on items).
  double throughput_weight_;

  mutable absl::Mutex mu_;

  // Current recommendation returned by `GetMaxChunkLength`. Will always be in
  // the range [1, num_keep_alive_refs_].
  int max_chunk_length_ ABSL_GUARDED_BY(mu_);

  // The most recent score which resulted in a change of `max_chunk_length_`. Is
  // initialized to {-1, -1} so an update of `max_chunk_length_` is triggered
  // regardless of what the first score is.
  struct Score {
    double average_chunk_length;
    double cost;
  };
  Score prev_score_ ABSL_GUARDED_BY(mu_);

  struct Statistic {
    // Key of the item or chunk.
    uint64_t key;
    // The average number of bytes sent per step. For items the number of steps
    // refers to the length of the item, not the length of the chunks
    // referenced. For chunks the length refers to the number of step
    // represented by the chunk.
    double bytes_per_step;
    // Average length of the chunks referenced by the item (or chunk).
    double average_chunk_length;
  };

  // Circular buffer of statistics of the `kNumItemsToScore` most recently
  // observed items.
  std::deque<Statistic> items_ ABSL_GUARDED_BY(mu_);

  // Circular buffer of statistics of the `kNumChunksToScore` most recently
  // observed chunks.
  std::deque<Statistic> chunks_ ABSL_GUARDED_BY(mu_);
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_CHUNKER_H_
