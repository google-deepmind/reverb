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

#ifndef REVERB_CC_REVERB_SERVER_H_
#define REVERB_CC_REVERB_SERVER_H_

#include <memory>

#include "reverb/cc/checkpointing/interface.h"
#include "reverb/cc/priority_table.h"
#include "reverb/cc/replay_client.h"
#include "reverb/cc/replay_service_impl.h"

namespace deepmind {
namespace reverb {

class ReverbServer {
 public:
  static tensorflow::Status StartReverbServer(
      std::vector<std::shared_ptr<PriorityTable>> priority_tables, int port,
      std::shared_ptr<CheckpointerInterface> checkpointer,
      std::unique_ptr<ReverbServer>* server);

  static tensorflow::Status StartReverbServer(
      std::vector<std::shared_ptr<PriorityTable>> priority_tables, int port,
      std::unique_ptr<ReverbServer>* server);

  ~ReverbServer();

  // Terminates the server and blocks until it has been terminated.
  void Stop();

  // Blocks until the server has terminated. Does not terminate the server
  // itself. Use this to want to wait indefinitely.
  void Wait();

  // Gets a local in process client. This bypasses proto serialization and
  // network overhead. Careful: The resulting client instance must not be used
  // after this server instance has terminated.
  std::unique_ptr<ReplayClient> InProcessClient();

 private:
  ReverbServer(std::vector<std::shared_ptr<PriorityTable>> priority_tables,
               int port,
               std::shared_ptr<CheckpointerInterface> checkpointer = nullptr);

  tensorflow::Status Initialize();

  // The port the server is on.
  int port_;

  std::unique_ptr<ReplayServiceImpl> replay_service_;

  std::unique_ptr<grpc::Server> server_ = nullptr;

  absl::Mutex mu_;
  bool running_ ABSL_GUARDED_BY(mu_) = false;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_REVERB_SERVER_H_
