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

#ifndef REVERB_CC_PLATFORM_SERVER_H_
#define REVERB_CC_PLATFORM_SERVER_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "reverb/cc/checkpointing/interface.h"
#include "reverb/cc/client.h"
#include "reverb/cc/table.h"

namespace deepmind {
namespace reverb {

// Unlimited.
constexpr int kMaxMessageSize = -1;

class Server {
 public:
  virtual ~Server() = default;

  // Terminates the server and blocks until it has been terminated.
  virtual void Stop() = 0;

  // Blocks until the server has terminated. Does not terminate the server
  // itself. Use this to want to wait indefinitely. Returns true if the server
  // was stopped by a SIGINT signal.
  virtual bool Wait() = 0;

  // Returns a summary string description.
  virtual std::string DebugString() const = 0;
};

absl::Status StartServer(std::vector<std::shared_ptr<Table>> tables, int port,
                         std::shared_ptr<Checkpointer> checkpointer,
                         std::unique_ptr<Server> *server);

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_PLATFORM_SERVER_H_
