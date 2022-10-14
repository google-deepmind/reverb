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

#include "reverb/cc/platform/thread.h"

#include <memory>
#include <thread>  // NOLINT(build/c++11)

#include "absl/memory/memory.h"

namespace deepmind {
namespace reverb {
namespace internal {
namespace {

class StdThread : public Thread {
 public:
  explicit StdThread(std::function<void()> fn) : thread_(std::move(fn)) {}

  ~StdThread() override { thread_.join(); }

 private:
  std::thread thread_;
};

}  // namespace

std::unique_ptr<Thread> StartThread(absl::string_view name,
                                    std::function<void()> fn) {
  return {std::make_unique<StdThread>(std::move(fn))};
}

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind
