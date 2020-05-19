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
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
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
  // The client must not be deleted while any of its writer instances exist.
  Writer(std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
         int chunk_length, int max_timesteps, bool delta_encoded = false,
         std::shared_ptr<internal::FlatSignatureMap> signatures = nullptr);
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
  // calling this method.
  tensorflow::Status Close();

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
  tensorflow::Status Finish();

  // Retries `WritePendingData` until sucessful or until non transient errors
  // encountered.
  tensorflow::Status WriteWithRetries();

  // Streams the chunks in `chunks_` referenced by `pending_items_` followed by
  // items in `pending_items_`
  bool WritePendingData();

  // Helper for generating a random ID.
  uint64_t NewID();

  // gRPC stub for the ReverbService.
  std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub_;

  // gRPC stream to the ReverbService.InsertStream endpoint.
  std::unique_ptr<grpc::ClientWriterInterface<InsertStreamRequest>> stream_;
  std::unique_ptr<grpc::ClientContext> context_;
  InsertStreamResponse response_;

  // The number of timesteps to batch in each chunk.
  const int chunk_length_;

  // The maximum number of recent timesteps which new items can reference.
  const int max_timesteps_;

  // Whether chunks should be delta encoded before compressed.
  const bool delta_encoded_;

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
