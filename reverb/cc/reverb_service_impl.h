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

#ifndef REVERB_CC_REVERB_SERVICE_IMPL_H_
#define REVERB_CC_REVERB_SERVICE_IMPL_H_

#include <memory>

#include "grpcpp/grpcpp.h"
#include "absl/numeric/int128.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "reverb/cc/checkpointing/interface.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/table.h"

namespace deepmind {
namespace reverb {

// Implements ReverbService. See reverb_service.proto for documentation.
class ReverbServiceImpl : public /* grpc_gen:: */ReverbService::Service {
 public:
  static tensorflow::Status Create(std::vector<std::shared_ptr<Table>> tables,
                                   std::shared_ptr<Checkpointer> checkpointer,
                                   std::unique_ptr<ReverbServiceImpl>* service);

  static tensorflow::Status Create(std::vector<std::shared_ptr<Table>> tables,
                                   std::unique_ptr<ReverbServiceImpl>* service);

  grpc::Status Checkpoint(grpc::ServerContext* context,
                          const CheckpointRequest* request,
                          CheckpointResponse* response) override;

  grpc::Status InsertStream(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<InsertStreamResponse, InsertStreamRequest>*
          stream) override;

  grpc::Status InsertStreamInternal(
      grpc::ServerContext* context,
      grpc::ServerReaderWriterInterface<InsertStreamResponse,
                                        InsertStreamRequest>* stream);

  grpc::Status MutatePriorities(grpc::ServerContext* context,
                                const MutatePrioritiesRequest* request,
                                MutatePrioritiesResponse* response) override;

  grpc::Status Reset(grpc::ServerContext* context, const ResetRequest* request,
                     ResetResponse* response) override;

  grpc::Status SampleStream(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<SampleStreamResponse, SampleStreamRequest>*
          stream) override;

  grpc::Status SampleStreamInternal(
      grpc::ServerContext* context,
      grpc::ServerReaderWriterInterface<SampleStreamResponse,
                                        SampleStreamRequest>* stream);

  grpc::Status ServerInfo(grpc::ServerContext* context,
                          const ServerInfoRequest* request,
                          ServerInfoResponse* response) override;

  grpc::Status InitializeConnection(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<InitializeConnectionResponse,
                               InitializeConnectionRequest>* stream) override;

  // Gets a copy of the table lookup.
  internal::flat_hash_map<std::string, std::shared_ptr<Table>> tables() const;

  // Closes all tables and the chunk store.
  void Close();

 private:
  explicit ReverbServiceImpl(
      std::shared_ptr<Checkpointer> checkpointer = nullptr);

  tensorflow::Status Initialize(std::vector<std::shared_ptr<Table>> tables);

  // Lookups the table for a given name. Returns nullptr if not found.
  Table* TableByName(absl::string_view name) const;

  // Checkpointer used to restore state in the constructor and to save data
  // when `Checkpoint` is called. Note that if `checkpointer_` is nullptr then
  // `Checkpoint` will return an `InvalidArgumentError`.
  std::shared_ptr<Checkpointer> checkpointer_;

  // Stores chunks and keeps references to them.
  ChunkStore chunk_store_;

  // Priority tables. Must be destroyed after `chunk_store_`.
  internal::flat_hash_map<std::string, std::shared_ptr<Table>> tables_;

  absl::BitGen rnd_;

  // A new id must be generated whenever a table is added, deleted, or has its
  // signature modified.
  absl::uint128 tables_state_id_;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_REVERB_SERVICE_IMPL_H_
