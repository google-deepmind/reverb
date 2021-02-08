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

#ifndef REVERB_CC_TRAJECTORY_WRITER_H_
#define REVERB_CC_TRAJECTORY_WRITER_H_

#include <deque>
#include <memory>
#include <vector>

#include "grpcpp/impl/codegen/client_context.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/signature.h"
#include "tensorflow/core/framework/tensor.h"

namespace deepmind {
namespace reverb {

// TODO(b/178096736): Write high level API documentation with examples.

class CellRef;
class Chunker;

// With the exception of `Close`, none of the methods are thread safe.
//
// TODO(b/178096736): Write high level API documentation with examples.
class TrajectoryWriter {
 public:
  struct Options {
    // The maximum size of the `Chunker` buffer. If it reaches this size then
    // its content is automatically finalized as a `ChunkData` and pushed to
    // related `CellRef`.
    int max_chunk_length;

    // The number of `CellRef` from most recent `Chunker::Append` calls to keep
    // alive. When a `CellRef` is removed from the buffer it can no longer be
    // referenced in new trajectories.
    //
    // Note that each column has its own buffer. This mean that if `Append`
    // calls do not provide data for all columns then the references will
    // expire with different `TrajectoryWriter::Append` calls.
    int num_keep_alive_refs;

    // Checks that field values are valid and returns `InvalidArgument` if any
    // field value, or combination of field values, are invalid.
    absl::Status Validate() const;
  };

  // TODO(b/178084425): Allow chunking options to be specified for each column.
  // TODO(b/178085651): Support initiation using the table signature.
  explicit TrajectoryWriter(
      std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
      const Options& options);

  // Flushes pending items and then closes stream. If `Close` has already been
  // called then no action is taken.
  ~TrajectoryWriter();

  // For each provided element in data (i.e not absl::nullopt), appends value to
  // corresponding column `Chunker`.
  //
  // If it is the first time data is provided for a column then a `Chunker` is
  // created and the dtype and shape of the tensor is used to build the column
  // spec. This spec is used to validate all successive attempts to append to
  // the column.
  //
  // `refs` will be grown to the same size as `data` (assuming that an empty
  // vector is provided). Indices in `data` where a value is set, a reference to
  // the value is given. The remaining elements will hold `absl::nullopt`. The
  // references should be used to define the trajectory in `CreateItem`.
  //
  // TODO(b/178085792): Figure out how episode information should be handled.
  // TODO(b/178085755): Decide how to manage partially invalid data.
  absl::Status Append(
      std::vector<absl::optional<tensorflow::Tensor>> data,
      std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Defines an item representing the data of `trajectory` and enques it for
  // insertion into `table` where it can be sampled according to `priority`.
  //
  // Before creating the item, `trajectory` is validated. A valid trajectory
  // must only use references to "live" data (i.e not yet expired due to
  // `num_keep_alive_refs`) created through `Append` calls on the same
  // `TrajectoryWriter` object. Furthermore, all `CellRef`s within each column
  // need to be compatible with each other. That is, they must have the same
  // dtype and have compatible shapes. If the trajectory is invalid then
  // `InvalidArgumentError` is returned.
  //
  // Note that this method will not block and wait for the IO to complete. This
  // means that if only `Append` and `CreateItem` are used then the caller will
  // not be impacted by the rate limiter on the server. Furthermore, the buffer
  // of pending items (and referenced data) could grow until the process runs
  // out of memory. The caller must therefore use `Flush` to achieve the
  // desired level of synchronization.
  absl::Status CreateItem(
      absl::string_view table, double priority,
      const std::vector<std::vector<std::weak_ptr<CellRef>>>& trajectory)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Sends all but the last `ignore_last_num_items` pending items and awaits
  // confirmation. Incomplete chunks referenced by these items are finalized
  // and transmitted.
  //
  // `ignore_last_num_items` can be used to limit how much the writer runs ahead
  // of the server, only blocking when the gap grows too big. For example, to
  // limit the "run ahead" to 20 just call `Flush(20)` after every `CreateItem`
  // call. If the number of unconfirmed items never reaches 20 (which is likely
  // if the rate limiter does not block), then no blocking ever occur. However,
  // if the sample rate suddenly falls and the rate limiter kicks in then the
  // `Flush` call blocks the writer from running ahead too much.
  absl::Status Flush(int ignore_last_num_items = 0,
                     absl::Duration timeout = absl::InfiniteDuration())
      ABSL_LOCKS_EXCLUDED(mu_);

  // Finalizes all chunks (including ones not referenced by any items), writes
  // and confirms all pending items, and resets the episode state (i.e generates
  // a new episode ID and sets step index to 0). If `clear_buffers` is true then
  // all `CellRef`s are invalidated (and their data deleted).
  absl::Status EndEpisode(
      bool clear_buffers, absl::Duration timeout = absl::InfiniteDuration());

  // Closes the stream, joins the worker thread and unblocks any concurrent
  // `Flush` call. All future (and concurrent) calls returns CancelledError once
  void Close() ABSL_LOCKS_EXCLUDED(mu_);

  // Attempts to configure a column `Chunker` (see `Chunker::Configure` for
  // details). If no `Chunker` exists for the column then the options will be
  // used to create the chunker when the column is present for the first time
  // in the data of an `Append` call.
  absl::Status ConfigureChunker(int column, const Options& options);

 private:
  using InsertStream = grpc::ClientReaderWriterInterface<InsertStreamRequest,
                                                         InsertStreamResponse>;

  struct ItemAndRefs {
    PrioritizedItem item;

    // Data referenced by the item. Note that the shared_ptr ensures that the
    // underlying data is not prematurely cleaned up even if it exceeds the max
    // age of the parent `Chunker`.
    std::vector<std::shared_ptr<CellRef>> refs;
  };

  // Sends all but the last `ignore_last_num_items` pending items and awaits
  // confirmation. Incomplete chunks referenced by non ignored items are
  // finalized and transmitted.
  absl::Status FlushLocked(int ignore_last_num_items, absl::Duration timeout)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Creates a gRPC stream to the server with `context_` and continues to run
  // until `closed_` set or until an error is encountered. In both cases
  // `Finish` is called on the stream and the status returned to the caller.
  //
  // Note that this method does not retry on any type of error status. Transient
  // errors are instead retried through resetting of `context_` and calling this
  // method again. This is managed by the anonymous function executed
  // by `worker_thread_`.
  absl::Status RunStreamWorker();

  // Sets `context_` and opens a gRPC InsertStream to the server.
  std::unique_ptr<InsertStream> SetContextAndCreateStream()
      ABSL_LOCKS_EXCLUDED(mu_);

  // Blocks until `write_queue_` is non-empty then copies the front element into
  // `item_and_refs`. If `Close` called before operation could complete, `false`
  // is returned.
  bool GetNextPendingItem(ItemAndRefs* item_and_refs) const
      ABSL_LOCKS_EXCLUDED(mu_);

  // Build and write the item insertion request to the stream. All chunks
  // referenced by item must have been written to the stream before calling this
  // method.
  bool SendItem(InsertStream* stream,
                const internal::flat_hash_set<uint64_t>& keep_keys,
                const PrioritizedItem& item) const;

  // Union of `GetChunkKeys` from all column chunkers and all the chunks
  // referenced by pending items (except for chunks only referenced by the first
  // item) filtered by presense in `streamed_chunk_keys. The chunks referenced
  // only by the first item can safely be ignored as the server "keep keys" is
  // updated with the insert item message.
  internal::flat_hash_set<uint64_t> GetKeepKeys(
      const internal::flat_hash_set<uint64_t>& streamed_chunk_keys) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Stub used to create InsertStream gRPC streams.
  std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub_;

  // Configuration options.
  Options options_;

  // Override of default options for yet to be constructed chunkers.
  internal::flat_hash_map<int, Options> options_override_;

  // Mapping from column index to Chunker. Shared pointers are used as the
  // `CellRef`s created by the chunker will own a weak_ptr created using
  // `weak_from_this()` on the Chunker.
  internal::flat_hash_map<int, std::shared_ptr<Chunker>> chunkers_;

  mutable absl::Mutex mu_;

  // ID of the active episode.
  uint64_t episode_id_ ABSL_GUARDED_BY(mu_);

  // Step within the episode.
  int episode_step_ ABSL_GUARDED_BY(mu_);

  // True if `Close` has been called.
  bool closed_ ABSL_GUARDED_BY(mu_);

  // Set if a non transient error encountered by the stream worker or if `Close`
  // has been called. In the latter case `unrecoverable_status_` will be set to
  // `CancelledError`.
  absl::Status unrecoverable_status_ ABSL_GUARDED_BY(mu_);

  // Items waiting for `stream_worker_` to write it to the steam.
  std::deque<ItemAndRefs> write_queue_ ABSL_GUARDED_BY(mu_);

  // Keys of items which have been written to the stream but for which no
  // confirmation has yet been received from the server.
  internal::flat_hash_set<uint64_t> in_flight_items_ ABSL_GUARDED_BY(mu_);

  // We signal when a chunk is flushed in case the stream worker backed off due
  // to the front item of `write_queue_` referencing incomplete chunks.
  absl::CondVar data_cv_ ABSL_GUARDED_BY(mu_);

  // Context used to create (and cancel) the gRPC stream used in
  // `stream_worker_`. The worker creates the context before invoking
  // `RunStreamWorker`. The mutex protects against potential data races between
  // concurrent `Close` calls and creation of new streams.
  std::unique_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mu_);

  // Creates `context_` and calls `RunStreamWorker` until `Close` called or
  // until the stream returns a non transient error. In both cases
  // `unrecoverable_status_` is populated before the thread is joinable.
  std::unique_ptr<internal::Thread> stream_worker_;
};

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
                      std::weak_ptr<CellRef>* ref)
      ABSL_LOCKS_EXCLUDED(mu_);

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
  // `Flush` just called.). Returns `InvalidArgumentError` if `max_chunk_length
  // > num_keep_alive_refs`  or if either is <= 0.
  absl::Status ApplyConfig(int max_chunk_length, int num_keep_alive_refs)
      ABSL_LOCKS_EXCLUDED(mu_);

 private:
  friend CellRef;

  // Get the data for referenced by `ref`. If the data has been finalized into
  // a ChunkData then the chunk is unpacked and the row extracted. If the chunk
  // has not been finalized the data is copied from `buffer_`.
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

  // Circular buffer of `CellRef`s that can be referenced in by new items. When
  // the size exceeds `num_keep_alive_refs_` then the oldest item is removed.
  std::deque<std::shared_ptr<CellRef>> active_refs_ ABSL_GUARDED_BY(mu_);
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_TRAJECTORY_WRITER_H_
