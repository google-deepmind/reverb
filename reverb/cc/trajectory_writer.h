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
#include <optional>
#include <string_view>
#include <vector>

#include "grpcpp/impl/codegen/client_context.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/signature.h"
#include "tensorflow/core/framework/tensor.h"

namespace deepmind {
namespace reverb {

// TODO(b/178096736): Write high level API documentation with examples.

class TrajectoryColumn;

// With the exception of `Close`, none of the methods are thread safe.
//
// TODO(b/178096736): Write high level API documentation with examples.
class TrajectoryWriter {
 public:
  struct Options {
    std::shared_ptr<ChunkerOptions> chunker_options;

    // Optional mapping from table names to optional flattened signatures. The
    // two layers of optional allows us to distinguish between tables that we
    // know exist but no signature is specified and a server were we have no
    // knowledge whatsoever about the tables.
    absl::optional<internal::FlatSignatureMap> flat_signature_map =
        absl::nullopt;

    // Checks that field values are valid and returns `InvalidArgument` if
    // any field value, or combination of field values, are invalid.
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

  // Same as `Append` but does not increment the episode step counter after
  // data has been appended to chunkers. This can be used when items need to
  // be created before all parts of the step structure is available (e.g learn
  // from the observation to select the next action in an on policy RL agent).
  //
  // One or more `AppendPartial` calls can be chained together as long as the
  // same column does not appear on more than one call. `Append` must be used to
  // add the final part of the step. If any column appears multiple times,
  // either in any of the `AppendPartial` calls or in the final `Append` call,
  // then `FailedPreconditionError` is returned.
  //
  // TODO(b/178085755): Decide how to manage partially invalid data.
  absl::Status AppendPartial(
      std::vector<absl::optional<tensorflow::Tensor>> data,
      std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Defines an item representing the data of `trajectory` and enques it for
  // insertion into `table` where it can be sampled according to `priority`.
  //
  // Before creating the item, `trajectory` is validated. A valid trajectory
  // must only use references to "live" data (i.e not yet expired due to
  // `num_keep_alive_refs`) created through `Append` calls on the same
  // `TrajectoryWriter` object. Furthermore, all `CellRef`s within each
  // column need to be compatible with each other. That is, they must have
  // the same dtype and have compatible shapes. If the trajectory is invalid
  // then `InvalidArgumentError` is returned.
  //
  // Note that this method will not block and wait for the IO to complete.
  // This means that if only `Append` and `CreateItem` are used then the
  // caller will not be impacted by the rate limiter on the server.
  // Furthermore, the buffer of pending items (and referenced data) could
  // grow until the process runs out of memory. The caller must therefore
  // use `Flush` to achieve the desired level of synchronization.
  absl::Status CreateItem(absl::string_view table, double priority,
                          absl::Span<const TrajectoryColumn> trajectory)
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
  absl::Status ConfigureChunker(int column,
                                const std::shared_ptr<ChunkerOptions>& options);

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

  // See `Append` and `AppendPartial`.
  absl::Status AppendInternal(
      std::vector<absl::optional<tensorflow::Tensor>> data,
      bool increment_episode_step,
      std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs)
      ABSL_LOCKS_EXCLUDED(mu_);

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

  // Checks that the table the item targets exists and that the item trajectory
  // conforms to the table signature (if any). Returns `InvalidArgumentError` if
  // the item is not valid.
  absl::Status Validate(const ItemAndRefs& item_and_refs) const;

  // Stub used to create InsertStream gRPC streams.
  std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub_;

  // Configuration options.
  Options options_;

  // Override of default options for yet to be constructed chunkers.
  internal::flat_hash_map<int, std::shared_ptr<ChunkerOptions>>
      options_override_;

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

class TrajectoryColumn {
 public:
  TrajectoryColumn(std::vector<std::weak_ptr<CellRef>> refs, bool squeeze);

  // Checks that the column is valid, if not returns `InvalidArgumentError`.
  //
  // A `TrajectoryColumns` is valid iff:
  //  * None of the `CellRef`s have expired.
  //  * All of the `CellRef`s have the same spec.
  //  * Columns has exactly one row if `squeeze` is true.
  //
  absl::Status Validate() const;

  // Locks and pushes all referenses to `locked_refs`.
  ABSL_MUST_USE_RESULT bool LockReferences(
      std::vector<std::shared_ptr<CellRef>>* locked_refs) const;

  // Encode as FlatTrajectory::Column proto.
  void ToProto(FlatTrajectory::Column* proto) const;

  // True if the column is empty.
  bool empty() const { return refs_.empty(); }

 private:
  // References to the rows that make up the column.
  std::vector<std::weak_ptr<CellRef>> refs_;

  // If set then the batch dimension is emitted when column is unpacked. Can
  // only be set when `refs_.size() == 1`.
  bool squeeze_;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_TRAJECTORY_WRITER_H_
