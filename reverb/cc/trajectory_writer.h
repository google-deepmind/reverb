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

#include <cstdint>
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
#include "reverb/cc/support/key_generators.h"
#include "reverb/cc/support/signature.h"
#include "tensorflow/core/framework/tensor.h"

namespace deepmind {
namespace reverb {

class TrajectoryColumn;   // Defined below.
class ArenaOwnedRequest;  // Defined in trajectory_writer.cc.

// A `ColumnWriter` allows creating replay items based on sparse or partial
// trajectories. The easiest way to explain a `ColumnWriter` is by comparing it
// to a the more traditional `Writer` (see writer.h).
//
// Think of a trajectory as a table or rows and columns. Rows are individual
// time steps and and columns of a row correspond to the tensors associated with
// each individual time step. The following table corresponds to a trajectory of
// length 4 (i.e. there are 4 steps in the trajectory) and each step can hold
// 5 tensors, which is why there are 5 columns.
//
//            C0   C1   C2   C3   C4
//          +----+----+----+----+----+
// Step 0   |    |    |    |    |    |
//          +----+----+----+----+----+
// Step 1   |    |    |    |    |    |
//          +----+----+----+----+----+
// Step 2   |    |    |    |    |    |
//          +----+----+----+----+----+
// Step 3   |    |    |    |    |    |
//          +----+----+----+----+----+
//
// When using the conventional `Writer`, we have to write dense rows where each
// cell in a row (i.e. the value of all columns) must be specified (and have the
// same spec as other values in the column). When creating priority items using
// the conventional `Writer`, we select a continuous range of (dense) rows and
// associate them with the new item. The following table shows a densely filled
// trajectory (a filled cell is marked with XX) and two priority items "Item0"
// and "Item1" that span the first 3 and last 3 steps of the trajectory
// respectively.
//
//            C0   C1   C2   C3   C4
//          +----+----+----+----+----+   --+--
// Step 0   | XX | XX | XX | XX | XX |   I |
//          +----+----+----+----+----+   t |    --+--
// Step 1   | XX | XX | XX | XX | XX |   e |    I |
//          +----+----+----+----+----+   m |    t |
// Step 2   | XX | XX | XX | XX | XX |   0 |    e |
//          +----+----+----+----+----+   --+--  m |
// Step 3   | XX | XX | XX | XX | XX |          1 |
//          +----+----+----+----+----+          --+--
//
// In contrast a `ColumnWriter` allows writing sparse rows and create sparse or
// non-continuous priority items. When writing a row using a `ColumnWriter`, we
// can leave the value of some columns unspecified. For example, we may only
// store values in columns C0, C1, and C4 and leave columns C3 and C4
// blank at one step and store values in all columns at another step.
//
//            C0   C1   C2   C3   C4
//          +----+----+----+----+----+
// Step 0   | XX | XX |    |    | XX |
//          +----+----+----+----+----+
// Step 1   | XX | XX | XX | XX | XX |
//          +----+----+----+----+----+
// Step 2   | XX | XX |    |    |    |
//          +----+----+----+----+----+
// Step 3   | XX | XX |    | XX |    |
//          +----+----+----+----+----+
//
// This is useful if we only actually consume data from some time steps. For
// example if we have two columns to store RNN states and observations, we may
// want to store  observation at every step of an episode but we may want to
// store the RNN state on the first step of an episode only. This can reduce
// memory usage as we don't need to store RNN states, which we don't consume.
//
// When creating a priority item using a `ColumnWriter`, we don't need to select
// a continuous range of rows. Instead, we may construct priority items that
// reference arbitrary cells of the episode table. For example, we can create a
// priority item, which references the first RNN state and every second
// observation (this specific example may not be of practical use, but it
// illustrates the point). In the following table "Item0" spans steps 0, 1, and
// 3 (skipping step 2) and Item1 spans the last 3 steps of the trajectory. In
// both items, the values stored per column varies.
//
//            C0   C1   C2   C3   C4
//          +----+----+----+----+----+  --+--
// Step 0   | XX | XX |    |    | XX |  I |
//          +----+----+----+----+----+  t |    --+--
// Step 1   | XX | XX | XX | XX | XX |  e |    I |
//          +----+----+----+----+----+  m |    t |
// Step 2   | XX | XX |    |    |    |    :    e |
//          +----+----+----+----+----+    :    m |
// Step 3   | XX | XX |    | XX |    |  0 |    1 |
//          +----+----+----+----+----+  --+--  --+--
//
//
// Reverb offers the following `ColumnWriter` implementations:
// - `TrajectoryWriter`: A fail-safe, asynchronous implementation. It ensures
//   handles all network activity on a background thread, has a non-blocking
//   API, and can recover from network failures without data loss.
// - `StreamingTrajectoryWriter`: An implementation that greedily streams data
//   to the replay server in an effort to reduce memory consumption on the
//   writing side. This writer is only useful when writing large amounts of data
//   using many threads on a single host. Data may be lost if network
//   connections fail and the API can be blocking as network writes are
//   performed on the calling thread. This is
// - `QueuetWriter`: based on the trajectory writer, it puts the trajectories
//    a queue without sending them to any server. It is used by the
//   `ReverbPatternDataset` to produce datasets of trajectories.

class ColumnWriter {
 public:
  virtual ~ColumnWriter() = default;

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
  virtual absl::Status Append(
      std::vector<absl::optional<tensorflow::Tensor>> data,
      std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs) = 0;

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
  virtual absl::Status AppendPartial(
      std::vector<absl::optional<tensorflow::Tensor>> data,
      std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs) = 0;

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
  virtual absl::Status CreateItem(
      absl::string_view table, double priority,
      absl::Span<const TrajectoryColumn> trajectory) = 0;

  // Finalizes all chunks (including ones not referenced by any items), writes
  // and confirms all pending items, and resets the episode state (i.e generates
  // a new episode ID and sets step index to 0). If `clear_buffers` is true then
  // all `CellRef`s are invalidated (and their data deleted).
  virtual absl::Status EndEpisode(
      bool clear_buffers,
      absl::Duration timeout = absl::InfiniteDuration()) = 0;

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
  virtual absl::Status Flush(
      int ignore_last_num_items = 0,
      absl::Duration timeout = absl::InfiniteDuration()) = 0;
};

// With the exception of `Close`, none of the methods are thread safe.
class TrajectoryWriter : public ColumnWriter,
                         public grpc::ClientBidiReactor<InsertStreamRequest,
                                                        InsertStreamResponse> {
 public:
  // Multiple `ChunkData` can be sent with the same `InsertStreamRequest`. If
  // the size of the message exceeds this value then the request is sent and the
  // remaining chunks are sent with other messages.
  static constexpr int64_t kMaxRequestSizeBytes = 40 * 1024 * 1024;  // 40MB.

  struct Options {
    // Checks that field values are valid and returns `InvalidArgument` if
    // any field value, or combination of field values, are invalid.
    absl::Status Validate() const;

    std::shared_ptr<ChunkerOptions> chunker_options;

    // Optional mapping from table names to optional flattened signatures. The
    // two layers of optional allows us to distinguish between tables that we
    // know exist but no signature is specified and a server were we have no
    // knowledge whatsoever about the tables.
    absl::optional<internal::FlatSignatureMap> flat_signature_map =
        absl::nullopt;
  };

  struct ItemAndRefs {
    // Checks that the table the item targets exists and that the item
    // trajectory conforms to the table signature (if any). Returns
    // `InvalidArgumentError` if the item is not valid.
    absl::Status Validate(const Options& options) const;

    PrioritizedItem item;

    // Data referenced by the item. Note that the shared_ptr ensures that the
    // underlying data is not prematurely cleaned up even if it exceeds the max
    // age of the parent `Chunker`.
    std::vector<std::shared_ptr<CellRef>> refs;
  };

  // TODO(b/178084425): Allow chunking options to be specified for each column.
  // TODO(b/178085651): Support initiation using the table signature.
  explicit TrajectoryWriter(
      std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
      const Options& options);

  // Flushes pending items and then closes stream. If `Close` has already been
  // called then no action is taken.
  ~TrajectoryWriter() override;

  // See `ColumnWriter::Append` above.
  absl::Status Append(std::vector<absl::optional<tensorflow::Tensor>> data,
                      std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs)
      override ABSL_LOCKS_EXCLUDED(mu_);

  // See `ColumnWriter::AppendPartial` above.
  absl::Status AppendPartial(
      std::vector<absl::optional<tensorflow::Tensor>> data,
      std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs) override
      ABSL_LOCKS_EXCLUDED(mu_);

  // See `ColumnWriter::CreateItem` above.
  absl::Status CreateItem(absl::string_view table, double priority,
                          absl::Span<const TrajectoryColumn> trajectory)
      override ABSL_LOCKS_EXCLUDED(mu_);

  // See `ColumnWriter::Flush` above.
  absl::Status Flush(int ignore_last_num_items = 0,
                     absl::Duration timeout = absl::InfiniteDuration()) override
      ABSL_LOCKS_EXCLUDED(mu_);

  // See `ColumnWriter::EndEpisode` above.
  absl::Status EndEpisode(
      bool clear_buffers,
      absl::Duration timeout = absl::InfiniteDuration()) override;

  // Closes the stream, joins the worker thread and unblocks any concurrent
  // `Flush` call. All future (and concurrent) calls returns CancelledError once
  void Close() ABSL_LOCKS_EXCLUDED(mu_);

  // Attempts to configure a column `Chunker` (see `Chunker::Configure` for
  // details). If no `Chunker` exists for the column then the options will be
  // used to create the chunker when the column is present for the first time
  // in the data of an `Append` call.
  absl::Status ConfigureChunker(int column,
                                const std::shared_ptr<ChunkerOptions>& options);

  // Get the maximum value for `keep_alive_refs` for any of the columns.
  int max_num_keep_alive_refs() const;

  // Number of `Append` calls since last `EndEpisode` call. Note that
  // `AppendPartial` calls does not increment this counter.
  int episode_steps() const;

  // Async GRPC callback handlers.
  void OnReadDone(bool ok) override;
  void OnWriteDone(bool ok) override;
  void OnDone(const ::grpc::Status& s) override;

 private:
  using InsertStream = grpc::ClientReaderWriterInterface<InsertStreamRequest,
                                                         InsertStreamResponse>;

  bool SendNotAlreadySentChunks(
      internal::flat_hash_set<uint64_t>* streamed_chunk_keys,
      absl::Span<const std::shared_ptr<CellRef>> refs,
      ArenaOwnedRequest* request);

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

  // Sets `context_` and opens a gRPC InsertStream to the server iff the writer
  // has not yet been closed.
  absl::Status SetContextAndCreateStream() ABSL_LOCKS_EXCLUDED(mu_);

  // Blocks until `write_queue_` is non-empty or the writer is being closed.
  // False is returned when writer should terminate without processing further
  // items.
  bool WaitForPendingItems() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Add an item to the insertion request. All chunks
  // referenced by item must have been written to the stream before calling this
  // method.
  void AddItemToRequest(const PrioritizedItem& item,
                        ArenaOwnedRequest* request);

  // Sends a given request to the server (if not empty). Tells server to keep
  // specified chunks for processing further requests.
  bool WriteIfNotEmpty(const internal::flat_hash_set<uint64_t>& keep_keys,
                       ArenaOwnedRequest* request) ABSL_LOCKS_EXCLUDED(mu_);

  // Terminates connection to the server.
  absl::Status Finish() ABSL_LOCKS_EXCLUDED(mu_);

  // Union of `GetChunkKeys` from all column chunkers and all the chunks
  // referenced by pending items (except for chunks only referenced by the first
  // item) filtered by presence in `streamed_chunk_keys. The chunks referenced
  // only by the first item can safely be ignored as the server "keep keys" is
  // updated with the insert item message.
  internal::flat_hash_set<uint64_t> GetKeepKeys(
      const internal::flat_hash_set<uint64_t>& streamed_chunk_keys) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Stub used to create InsertStream gRPC streams.
  std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub_;

  // Configuration options.
  Options options_;

  // Used to generates keys for episode and item IDs.
  std::unique_ptr<internal::KeyGenerator> key_generator_;

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
  std::deque<std::unique_ptr<ItemAndRefs>> write_queue_ ABSL_GUARDED_BY(mu_);

  // Items which have been written to the stream but for which no confirmation
  // has yet been received from the server. Note that we have to keep the item
  // alive until the confirmation has been received so that we are able to
  // retry the request if the server becomes unavailable.
  internal::flat_hash_map<uint64_t, std::unique_ptr<ItemAndRefs>> in_flight_items_
      ABSL_GUARDED_BY(mu_);

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

  // Response received from the server. It is only accessed by the onReadDone.
  InsertStreamResponse response_;

  // Is there currently an inflight request to the server.
  bool write_inflight_ ABSL_GUARDED_BY(mu_);

  // Is the current connection to the server terminated.
  bool stream_done_ ABSL_GUARDED_BY(mu_);

  // Is stream good for sending/receiving requests.
  bool stream_ok_ ABSL_GUARDED_BY(mu_);

  // In case `stream_done_` == false, tha status of the terminated connection.
  absl::Status stream_status_ ABSL_GUARDED_BY(mu_);
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

  // True if the column is squeezed.
  bool squeezed() const { return squeeze_; }

  // Returns a reference to the rows.
  std::vector<std::weak_ptr<CellRef>>& refs() { return refs_; }

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
