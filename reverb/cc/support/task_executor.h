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
// * TaskInfo is a class containing task information.
// * TaskCallback is a function with the following signature:
//   `void task_callback(TaskInfo task_info, const absl::Status&)`
//   This function will be called with the threads of the worker.
template <class TaskInfo, class TaskCallback>
class TaskExecutor {
 public:
  // Constructs a TaskExecutor.
  // num_threads: number of threads that will run tasks.
  // thread_name_prefix: is used as a prefix for the name of the threads.
  TaskExecutor(size_t num_threads, const std::string& thread_name_prefix);

  ~TaskExecutor();

  // Schedules `task_cb` to be called with `task_info` as soon as possible.
  void Schedule(TaskInfo task_info, TaskCallback task_cb);

  // Closes the thread pool and the queue. After calling this, no new tasks
  // will be scheduled and pending tasks will run with a cancelled status.
  void Close();

 private:
  struct Task {
    TaskCallback callback;
    TaskInfo task_info;
  };

  void RunWorker();

  internal::UnboundedQueue<Task> queue_;
  std::vector<std::unique_ptr<internal::Thread>> threads_;
};

/*****************************************************************************
 * Implementation of the template                                            *
 *****************************************************************************/

template <class TaskInfo, class TaskCallback>
TaskExecutor<TaskInfo, TaskCallback>::TaskExecutor(
    size_t num_threads, const std::string& thread_name_prefix)
    : queue_() {
  for (int thread_index = 0; thread_index < num_threads; thread_index++) {
    threads_.push_back(internal::StartThread(
        absl::StrCat(thread_name_prefix, "_", thread_index),
        [this] { RunWorker(); }));
  }
}

template <class TaskInfo, class TaskCallback>
TaskExecutor<TaskInfo, TaskCallback>::~TaskExecutor() {
  Close();
}

template <class TaskInfo, class TaskCallback>
void TaskExecutor<TaskInfo, TaskCallback>::Schedule(TaskInfo task_info,
                                                    TaskCallback task_cb) {
  queue_.Push({.callback = std::move(task_cb),
               .task_info = std::move(task_info)});
}

template <class TaskInfo, class TaskCallback>
void TaskExecutor<TaskInfo, TaskCallback>::Close() {
  Task task;
  // Before closing, we cancel all the pending tasks.
  queue_.SetLastItemPushed();
  while (queue_.Pop(&task)) {
    task.callback(std::move(task.task_info),
                  absl::CancelledError("Task queue is closed."));
  }
  threads_.clear();  // Joins worker threads.
}

template <class TaskInfo, class TaskCallback>
void TaskExecutor<TaskInfo, TaskCallback>::RunWorker() {
  Task task;
  while (queue_.Pop(&task)) {
    task.callback(std::move(task.task_info), absl::OkStatus());
  }
}

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_TASK_EXECUTOR_H_
