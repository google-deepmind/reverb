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
#include "tensorflow/core/platform/status.h"

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
  // references should be used to define the trajectory in `InsertItem`.
  //
  // TODO(b/178085792): Figure out how episode information should be handled.
  // TODO(b/178085755): Decide how to manage partially invalid data.
  tensorflow::Status Append(
      std::vector<absl::optional<tensorflow::Tensor>> data,
      std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Defines an item representing the data of `trajectory` and enques it for
  // insertion into `table` where it can be sampled according to `priority`.
  //
  // Before creating the item, `trajectory` is validated. A valid trajectory
  // must only use references to "live" data (i.e not yet expired due to
  // `num_keep_alive_refs`) created through `Append` calls on the same
  // `TrajectoryWriter` object. If the trajectory is invalid then
  // `InvalidArgumentError` is returned.
  //
  // Note that this method will not block and wait for the IO to complete. This
  // means that if only `Append` and `InsertItem` are used then the caller will
  // not be impacted by the rate limiter on the server. Furthermore, the buffer
  // of pending items (and referenced data) could grow until the process runs
  // out of memory. The caller must therefore use `Flush` to achieve the
  // desired level of synchronization.
  //
  // TODO(b/178089532): Validate that trajectory columns can be concatenated.
  tensorflow::Status InsertItem(
      absl::string_view table, double priority,
      const std::vector<std::vector<std::weak_ptr<CellRef>>>& trajectory)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Sends all pending items and awaits confirmation. Incomplete chunks
  // referenced by pending items are finalized and transmitted.
  //
  // TODO(b/178087048): Support flushing and blocking until at most N items are
  //   unconfirmed.
  tensorflow::Status Flush(absl::Duration timeout = absl::InfiniteDuration())
      ABSL_LOCKS_EXCLUDED(mu_);

  // TODO(b/178085792): Consider adding an EndEpisode method.

  // Closes the stream, joins the worker thread and unblocks any concurrent
  // `Flush` call. All future (and concurrent) calls returns CancelledError once
  void Close() ABSL_LOCKS_EXCLUDED(mu_);

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

  // Creates a gRPC stream to the server with `context_` and continues to run
  // until `closed_` set or until an error is encountered. In both cases
  // `Finish` is called on the stream and the status returned to the caller.
  //
  // Note that this method does not retry on any type of error status. Transient
  // errors are instead retried through resetting of `context_` and calling this
  // method again. This is managed by the anonymous function executed
  // by `worker_thread_`.
  tensorflow::Status RunStreamWorker();

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
                const internal::flat_hash_set<uint64_t>& streamed_chunk_keys,
                const PrioritizedItem& item) const;

  // Stub used to create InsertStream gRPC streams.
  std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub_;

  // Configuration options.
  Options options_;

  // Mapping from column index to Chunker.
  internal::flat_hash_map<int, std::unique_ptr<Chunker>> chunkers_;

  mutable absl::Mutex mu_;

  // True if `Close` has been called.
  bool closed_ ABSL_GUARDED_BY(mu_);

  // Set if a non transient error encountered by the stream worker or if `Close`
  // has been called. In the latter case `unrecoverable_status_` will be set to
  // `CancelledError`.
  tensorflow::Status unrecoverable_status_ ABSL_GUARDED_BY(mu_);

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
  CellRef(Chunker* chunker, uint64_t chunk_key, int offset);

  // Key of the parent chunk.
  uint64_t chunk_key() const;

  // Offset within the parent chunk.
  int offset() const;

  // True if SetChunk has been called.
  bool IsReady() const ABSL_LOCKS_EXCLUDED(mu_);

  // Gets chunker if set. If not yet set then nullptr is returned.
  std::shared_ptr<ChunkData> GetChunk() const ABSL_LOCKS_EXCLUDED(mu_);

 private:
  friend Chunker;

  // Called by chunker the referenced data is flushed into a ChunkData.
  void SetChunk(std::shared_ptr<ChunkData> chunk) ABSL_LOCKS_EXCLUDED(mu_);

 private:
  friend TrajectoryWriter;

  // Raw pointer to the parent. This method should only be used by the ancestor
  // `TrajectoryWriter` whos existence guarantees that the `Chunker` also exists
  // and thus the pointer can safely be dereferenced.
  Chunker* chunker() const;

 private:
  // Chunker which created the `CellRef` and will eventually create the chunk
  // and call `SetChunk`. Note that the `CellRef` may outlive the parent
  // `Chunker` so it is not generally safe to dereference this value.
  Chunker* chunker_;

  // Key of the parent chunk.
  uint64_t chunk_key_;

  // Offset of element within the parent chunk.
  int offset_;

  mutable absl::Mutex mu_;

  // Parent chunk which eventually be set by parent `Chunker`.
  absl::optional<std::shared_ptr<ChunkData>> chunk_ ABSL_GUARDED_BY(mu_);
};

class Chunker {
 public:
  Chunker(internal::TensorSpec spec, int max_chunk_length,
          int num_keep_alive_refs);

  // Validates `tensor` against `spec_`, appends it to the active chunk and
  // returns a reference to the new row. If the active chunk now has
  // `max_chunk_length`th rows then it is finalized and its `CellRef`s
  // notified (including the one returned by this call).
  tensorflow::Status Append(tensorflow::Tensor tensor,
                            std::weak_ptr<CellRef>* ref)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Creates a chunk from the data in the buffer and calls `SetChunk` on its
  // `CellRef`s.
  tensorflow::Status Flush() ABSL_LOCKS_EXCLUDED(mu_);

  // Keys of the FINALIZED chunks referenced by `CellRef`s in `active_refs_`.
  std::vector<uint64_t> GetKeepKeys() const ABSL_LOCKS_EXCLUDED(mu_);

 private:
  tensorflow::Status FlushLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

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
