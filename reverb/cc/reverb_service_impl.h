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

#ifndef REVERB_CC__REVERB_SERVICE_IMPL_H_
#define REVERB_CC__REVERB_SERVICE_IMPL_H_

#include <memory>

#include "grpcpp/grpcpp.h"
#include "absl/numeric/int128.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "reverb/cc/checkpointing/interface.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/table.h"

namespace deepmind {
namespace reverb {

// Implements ReverbService asynchronously. See reverb_service.proto for
// documentation.
class ReverbServiceImpl : public /* grpc_gen:: */ReverbService::CallbackService {
 public:
  static absl::Status Create(
      std::vector<std::shared_ptr<Table>> tables,
      std::shared_ptr<Checkpointer> checkpointer,
      std::unique_ptr<ReverbServiceImpl>* service);

  static absl::Status Create(
      std::vector<std::shared_ptr<Table>> tables,
      std::unique_ptr<ReverbServiceImpl>* service);

  grpc::ServerUnaryReactor* Checkpoint(grpc::CallbackServerContext* context,
                                       const CheckpointRequest* request,
                                       CheckpointResponse* response) override;

  // The InsertStream call schedules items to be inserted and to send back
  // confirmation when requested.
  // 1. The Reactor starts waiting to read
  // 2. Once the read has arrived, OnReadDone either adds a chunk or schedules
  // an item to be inserted.
  // 3. After a read is done, it will start a new read if the number of pending
  // tasks in the InsertWorker is below max_queue_size_to_warn. Reads will be
  // resumed once an insertion task is completed if there are no reads
  // in-flight and there is enough free space in the InsertWorker queue.
  //
  // Once an insertion task is completed, the reactor writes confirmations of
  // inserted messages.
  //
  // Once the reactor is done, it waits for all the items to be inserted before
  // destroying itself. If the reactor is cancelled, pending items won't be
  // inserted but we still need to wait for the insert worker to dequeue them
  // from its tasks list.
  //
  // Note on blocking reads when the Worker queue is almost full:
  // 1. We cannot block a new reactor because it is unaware of the InsertWorker
  // state until it has scheduled its first insertion (after which, it will stor
  // reading until there is empty space).
  // 2. When reads are blocked, inserting new chunks gets blocked as well. This
  // is intended as it limits the memory usage when we have many (blocked)
  // connections and pending items to be inserted.
  // 3. When the last scheduled insertion runs, we reactivate the reads even if
  // the number of items in the queue exceeds max_queue_size_to_read as it is
  // the last opportunity we have to resume reads.
  grpc::ServerBidiReactor<InsertStreamRequest, InsertStreamResponse>*
  InsertStream(grpc::CallbackServerContext* context) override;

  grpc::ServerUnaryReactor* MutatePriorities(
      grpc::CallbackServerContext* context,
      const MutatePrioritiesRequest* request,
      MutatePrioritiesResponse* response) override;

  grpc::ServerUnaryReactor* Reset(grpc::CallbackServerContext* context,
                                  const ResetRequest* request,
                                  ResetResponse* response) override;

  // The SampleStream call preserves the behavior of the synchronous
  // implementation.
  // 1. The Reactor starts waiting for a request.
  // 2. Once the request has arrived, OnReadDone samples one batch and writes
  // one of the messages of the batch.
  // 3. Once a write finishes:
  //    3.a. if the batch of samples is complete, and all the requested data has
  //    been sent, starts a new Read.
  //    3.b. if the batch of samples is complete but
  //    the client requested more data, it samples another batch and writes the
  //    first message of the batch.
  //    3.c. if there are still messages to be sent,
  //    writes one message.
  grpc::ServerBidiReactor<SampleStreamRequest, SampleStreamResponse>*
  SampleStream(grpc::CallbackServerContext* context) override;

  grpc::ServerUnaryReactor* ServerInfo(grpc::CallbackServerContext* context,
                                       const ServerInfoRequest* request,
                                       ServerInfoResponse* response) override;

  grpc::ServerBidiReactor<InitializeConnectionRequest,
                          InitializeConnectionResponse>*
  InitializeConnection(grpc::CallbackServerContext* context) override;

  // Gets a copy of the table lookup.
  internal::flat_hash_map<std::string, std::shared_ptr<Table>> tables() const;

  // Closes all tables and the chunk store.
  void Close();

  // Returns a summary string description.
  std::string DebugString() const;

 private:
  explicit ReverbServiceImpl(
      std::shared_ptr<Checkpointer> checkpointer = nullptr);

  absl::Status Initialize(std::vector<std::shared_ptr<Table>> tables);

  // Lookups the table for a given name. Returns nullptr if not found.
  std::shared_ptr<Table> TableByName(absl::string_view name) const;

  // Checkpointer used to restore state in the constructor and to save data
  // when `Checkpoint` is called. Note that if `checkpointer_` is nullptr then
  // `Checkpoint` will return an `InvalidArgumentError`.
  std::shared_ptr<Checkpointer> checkpointer_;

  // Priority tables.
  internal::flat_hash_map<std::string, std::shared_ptr<Table>> tables_;

  absl::BitGen rnd_;

  // A new id must be generated whenever a table is added, deleted, or has its
  // signature modified.
  absl::uint128 tables_state_id_;
};


}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC__REVERB_SERVICE_IMPL_H_
