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

#ifndef LEARNING_DEEPMIND_REPLAY_REVERB_WRITER_H_
#define LEARNING_DEEPMIND_REPLAY_REVERB_WRITER_H_

#include <list>
#include <memory>
#include <vector>

#include "grpcpp/impl/codegen/client_context.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include <cstdint>
#include "absl/base/thread_annotations.h"
#include "absl/flags/flag.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/signature.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace deepmind {
namespace reverb {

// None of the methods are thread safe.
class Writer {
 public:
  static constexpr absl::optional<int> kDefaultMaxInFlightItems = absl::nullopt;

  // The client must not be deleted while any of its writer instances exist.
  Writer(std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
         int chunk_length, int max_timesteps, bool delta_encoded = false,
         std::shared_ptr<internal::FlatSignatureMap> signatures = nullptr,
         absl::optional<int> max_in_flight_items = absl::nullopt);
  ~Writer();

  // Appends a timestamp to internal `buffer_`. If the size of the buffer
  // reached `chunk_length_` then its content is batched and inserted into
  // `chunks_`. If `pending_items_` is not empty then its items are streamed to
  // the ReverbService and popped.
  //
  // If all operations are successful then `buffer_` is cleared, a new
  // `next_chunk_key_` is set and old items are removed from `chunks_` until its
  // size is <= `max_chunks_`. If unsuccessful all internal state is reverted.
  tensorflow::Status Append(std::vector<tensorflow::Tensor> data);

  // Appends a batched sequence of timesteps. Equivalent to calling `Append` `T`
  // times where `T` is batch size of `sequence`. The shapes of the elements of
  // `sequence` thus have `[T] + shape_of_single_timestep_element`.
  tensorflow::Status AppendSequence(std::vector<tensorflow::Tensor> sequence);

  // Adds a new PrioritizedItem to `table` spanning the last `num_timesteps` and
  // pushes new item to `pending_items_`. If `buffer_` is empty then the new
  // item is streamed to the ReverbService. If unsuccessful all internal state
  // is reverted.
  tensorflow::Status CreateItem(const std::string& table, int num_timesteps,
                                double priority);

  // TODO(b/154929199): There should probably be a method for ending an episode
  // even if you don't want to close the stream.

  // Creates a new batch from the content of `buffer_` and writes all
  // `pending_items_` and closes the stream_. The object must be abandoned after
  // calling this method. Iff `max_items_in_flight_` is set then call blocks
  // until server has confirmed that all items have been written.
  // If retry_on_unavailable is true, it will keep trying to send to the server
  // when the server returns Unavailable errors.
  tensorflow::Status Close(bool retry_on_unavailable = true);

  // Creates a new batch from the content of `buffer_` and writes all
  // `pending_items_`.  This is useful to force any pending items to be sent to
  // the replay buffer. Iff `max_items_in_flight_` is set then call blocks until
  // server has confirmed that all items have been written.

  // TODO(b/159623854): Add a configurable timeout and pipe it through to the
  // python API.
  tensorflow::Status Flush();

 private:
  // Creates a new batch from the content of `buffer_` and inserts it into
  // `chunks_`. If `pending_items_` is not empty then the items are streamed to
  // the ReverbService and popped.
  //
  // If all operations are successful then `buffer_` is cleared, a new
  // `next_chunk_key_` is set, `index_within_episode_` is incremented by the
  // number of items in `buffer_` and old items are removed from `chunks_` until
  // its size is <= `max_chunks_`. If the operation was unsuccessful then chunk
  // is popped from `chunks_`.
  tensorflow::Status Finish(bool retry_on_unavailable);

  // Retries `WritePendingData` until sucessful or, if retry_on_unavailable is
  // true, until non transient errors encountered
  tensorflow::Status WriteWithRetries(bool retry_on_unavailable);

  // Streams the chunks in `chunks_` referenced by `pending_items_` followed by
  // items in `pending_items_`
  bool WritePendingData();

  // Helper for generating a random ID.
  uint64_t NewID();

  // Blocks until the number of in flight items is <= `limit` or until reading
  // from the stream fails. Returns true if `limit` reached.
  bool ConfirmItems(int limit) ABSL_LOCKS_EXCLUDED(mu_);

  // Reads from `stream_` when `num_items_in_flight_` is > 0.
  //
  // Sets `item_confirmation_worker_running_` when setup and clears it just
  // before returning.
  //
  // Stops running if `item_confirmation_stop_requested_` set or if
  // `stream_->Read` fails.
  //
  // NOTE! `Start/StopItemConfirmationWorker` must be used to spawn and join
  // the worker thread to ensure that `stream_` is present during the entire
  // lifetime of the worker.
  void ItemConfirmationWorker() ABSL_LOCKS_EXCLUDED(mu_);

  // Spawns, sets and waits for `item_confirmation_worker_thread_` to start
  // running.
  void StartItemConfirmationWorker() ABSL_LOCKS_EXCLUDED(mu_);

  // Stops and joins `item_confirmation_worker_thread_`.
  //
  // NOTE! This method sets `item_confirmation_worker_stop_requested_` to force
  // the completion of the worker thread. It does not wait for items currently
  // in flight to be confirmed so unless this method is called as a result of an
  // error then `ConfirmItems(0)` must be called before invoking
  // `StopItemConfirmationWorker`.
  tensorflow::Status StopItemConfirmationWorker() ABSL_LOCKS_EXCLUDED(mu_);

  // gRPC stub for the ReverbService.
  std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub_;

  // gRPC stream to the ReverbService.InsertStream endpoint.
  std::unique_ptr<grpc::ClientReaderWriterInterface<InsertStreamRequest,
                                                    InsertStreamResponse>>
      stream_;
  std::unique_ptr<grpc::ClientContext> context_;

  // The number of timesteps to batch in each chunk.
  const int chunk_length_;

  // The maximum number of recent timesteps which new items can reference.
  const int max_timesteps_;

  // Whether chunks should be delta encoded before compressed.
  const bool delta_encoded_;

  // The maximum number if items that is allowed to be "in flight" (i.e sent to
  // the server but not yet confirmed to be completed) at the same time. If this
  // value is reached and an item is about to be sent then the operation will
  // block until the completion of a previously transmitted item has been
  // confirmed. When set to `absl::nullopt` the number of concurrent "in flight"
  // items is unlimited.
  const absl::optional<int> max_in_flight_items_;

  // Number of items that have been sent to the server but the response
  // confirming the completion of the operation haven't been read yet.
  int num_items_in_flight_ ABSL_GUARDED_BY(mu_);

  // Flag set by `item_confirmation_worker_thread_` (through
  // `ItemConfirmationWorker`) after it has stopped reading from the stream. The
  // flag allows threads that await `num_items_in_flight_` to reach some value
  // to be unblocked when the `item_confirmation_worker_thread_` exists
  // prematurely (i.e before all items have been confirmed) due to errors with
  // the stream.
  bool item_confirmation_worker_running_ ABSL_GUARDED_BY(mu_) = false;

  // Set by `StopItemConfirmationWorker` to wake up (and terminate) worker
  // thread even if no items are currently in flight.
  bool item_confirmation_worker_stop_requested_ ABSL_GUARDED_BY(mu_) = false;

  // Protects access to `num_items_in_flight_`,
  // `item_confirmation_worker_running_`,
  // `item_confirmation_worker_stop_requested_` and
  // `item_confirmation_worker_thread_`.
  absl::Mutex mu_;

  // Worker thread that reads item confirmations from the stream.
  std::unique_ptr<internal::Thread> item_confirmation_worker_thread_
      ABSL_GUARDED_BY(mu_) = nullptr;

  // Cache mapping table name to cached flattened signature.
  std::shared_ptr<internal::FlatSignatureMap> signatures_;

  // Bit generator used by `NewID`.
  absl::BitGen bit_gen_;

  // PriorityItems waiting to be sent to the ReverbService. Items are appended
  // to the list when they reference timesteps in `buffer_`. Once `buffer_` has
  // size `chunk_length_` the content is chunked and the pending items are
  // written to the ReverbService. While `buffer_` is empty new items are
  // written to the ReverbService immediately.
  std::list<PrioritizedItem> pending_items_;

  // Timesteps not yet batched up and put into `chunks_`.
  std::vector<std::vector<tensorflow::Tensor>> buffer_;

  // Batched timesteps that can be referenced by new items.
  std::list<ChunkData> chunks_;

  // Keys of the chunks which have been streamed to the server.
  absl::flat_hash_set<uint64_t> streamed_chunk_keys_;

  // The key used to reference the items currently in `buffer_`.
  uint64_t next_chunk_key_;

  // The episode id to attach to inserted timesteps.
  uint64_t episode_id_;

  // Index of the first timestep in `buffer_`.
  int32_t index_within_episode_;

  // Set if `Close` has been called.
  bool closed_;

  // Set of signatures passed to Append in a circular buffer.  Each
  // entry is the flat list of tensor dtypes and shapes in past Append
  // calls.  The vector itself is of length max_time_steps_ and Append
  // updates the DtypesAndShapes at index append_dtypes_and_shapes_location_.
  std::vector<internal::DtypesAndShapes> inserted_dtypes_and_shapes_;
  int insert_dtypes_and_shapes_location_ = 0;

  // Get a pointer to the DtypesAndShapes flattened signature for `table`.
  // Returns a nullopt signature if no signatures were provided to the Writer on
  // initialization.  Raises an InvalidArgument if the table is not in
  // signatures_.
  tensorflow::Status GetFlatSignature(
      absl::string_view table,
      const internal::DtypesAndShapes** dtypes_and_shapes) const;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // LEARNING_DEEPMIND_REPLAY_REVERB_WRITER_H_
