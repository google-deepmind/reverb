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

#ifndef REVERB_CC_THREAD_STATS_H_
#define REVERB_CC_THREAD_STATS_H_

#include <string>
#include <vector>

#include "absl/time/time.h"

namespace deepmind {
namespace reverb {

struct ThreadStats {
  // ThreadStats gathers stats about one TaskWorker thread in terms of the Task
  // that the thread is currently working on (and the ones it has worked on).

  // Sequence number of the current task being processed by this thread.
  // This number is incremented each time the thread picks a new task. If
  // current_task < num_tasks_processed, the thread is waiting for a new task.
  int current_task_id = -1;

  // Timestamp indicating when the task was inserted in the queue.
  absl::Time current_task_created_at;
  // Timestamp indicating when the thread started working on the current task.
  absl::Time current_task_started_at;
  // Information about the current task that can be used to identify the task
  // when printing the stats (e.g., the debug string of the task).
  std::string current_task_info = "";
  // Number of tasks that have been processed and completed. It doesn't include
  // the current task being processed.
  int num_tasks_processed = 0;
};

int LastThreadId(const std::vector<ThreadStats>& stats);

std::string FormatThreadStats(const std::vector<ThreadStats>& stats);
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_THREAD_STATS_H_
