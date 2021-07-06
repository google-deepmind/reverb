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

#include "reverb/cc/support/task_executor.h"

namespace deepmind {
namespace reverb {

TaskExecutor::TaskExecutor(size_t num_threads,
                           const std::string& thread_name_prefix)
    : queue_() {
  for (int thread_index = 0; thread_index < num_threads; thread_index++) {
    threads_.push_back(internal::StartThread(
        absl::StrCat(thread_name_prefix, "_", thread_index),
        [this] { RunWorker(); }));
  }
}

TaskExecutor::~TaskExecutor() {
  Close();
}

void TaskExecutor::Schedule(const std::function<void()>& callback) {
  queue_.Push(std::move(callback));
}

void TaskExecutor::Close() {
  std::function<void()> callback;
  // Before closing, we cancel all the pending tasks.
  queue_.SetLastItemPushed();
  while (queue_.Pop(&callback)) {
    callback();
  }
  threads_.clear();  // Joins worker threads.
}

void TaskExecutor::RunWorker() {
  std::function<void()> callback;
  while (queue_.Pop(&callback)) {
    callback();
  }
}

}  // namespace reverb
}  // namespace deepmind
