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

// Used to switch between different threading implementations. Concretely we
// switch between Google internal threading libraries and std::thread.

#ifndef REVERB_CC_SUPPORT_THREAD_H_
#define REVERB_CC_SUPPORT_THREAD_H_

#include <functional>
#include <memory>

#include "absl/strings/string_view.h"

namespace deepmind {
namespace reverb {
namespace internal {

// The `Thread` class can be subclassed to hold an object that invokes a
// method in a separate thread. A `Thread` is considered to be active after
// construction until the execution terminates. Calling the destructor of this
// class must join the separate thread and block until it has completed.
// Use `StartThread()` to create an instance of this class.
class Thread {
 public:
  // Joins the running thread, i.e. blocks until the thread function has
  // returned.
  virtual ~Thread() = default;

  // A Thread is not copyable.
  Thread(const Thread&) = delete;
  Thread& operator=(const Thread&) = delete;

 protected:
  Thread() = default;
};

// Starts a new thread that executes (a copy of) fn. The `name_prefix` may be
// used by the implementation to label the new thread.
std::unique_ptr<Thread> StartThread(absl::string_view name_prefix,
                                    std::function<void()> fn);

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_THREAD_H_
