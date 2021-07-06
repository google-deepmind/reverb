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

#ifndef REVERB_CC_TASK_EXECUTOR_H_
#define REVERB_CC_TASK_EXECUTOR_H_

#include <string>

#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/support/unbounded_queue.h"

namespace deepmind {
namespace reverb {

// Class that implements a thread pool that executes tasks from a queue. It is
// thread-safe.
class TaskExecutor {
 public:
  // Constructs a TaskExecutor.
  // num_threads: number of threads that will run tasks.
  // thread_name_prefix: is used as a prefix for the name of the threads.
  TaskExecutor(size_t num_threads, const std::string& thread_name_prefix);

  ~TaskExecutor();

  // Schedules `task_cb` to be called as soon as possible.
  void Schedule(const std::function<void()>& callback);

  // Closes the thread pool and the queue. After calling this, no new tasks
  // will be scheduled and pending tasks will run with a cancelled status.
  void Close();

 private:
  void RunWorker();

  internal::UnboundedQueue<std::function<void()>> queue_;
  std::vector<std::unique_ptr<internal::Thread>> threads_;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_TASK_EXECUTOR_H_
