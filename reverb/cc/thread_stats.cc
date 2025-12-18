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

#include "reverb/cc/thread_stats.h"

#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace deepmind {
namespace reverb {
// Returns the id of the worker thread that was the last one that picked up a
// new task. Returns -1 if any thread is waiting for a new task.
int LastThreadId(const std::vector<ThreadStats>& stats) {
  auto last_task_started = absl::InfinitePast();
  int index = 0;
  for (int i = 0; i < stats.size(); i++) {
    // current_task_id is local to the thread_stats and increments every time
    // the thread starts a new task (the initial value is -1).
    // num_tasks_processed gets incremented once the task is done (the initial
    // value is 0).
    // While the thread is busy with a task, current_task_id ==
    // num_tasks_processed,
    if (stats[i].current_task_id < stats[i].num_tasks_processed ||
        stats[i].current_task_id == -1) {
      // The current task is already processed (or the thread did not pick any
      // task yet), so this thread is free to start a new task.)
      return -1;
    }
    if (last_task_started < stats[i].current_task_started_at) {
      last_task_started = stats[i].current_task_started_at;
      index = i;
    }
  }
  return index;
}

std::string FormatThreadStats(const std::vector<ThreadStats>& stats) {
  std::string s = "";
  for (int i = 0; i < stats.size(); i++) {
    absl::StrAppendFormat(&s, "\tThread[%d]:\n", i);
    absl::StrAppendFormat(&s, "\t\tcurently processing task: %d (info: %s)\n",
                          stats[i].current_task_id, stats[i].current_task_info);
    absl::StrAppendFormat(&s, "\t\tTotal number of tasks processed: %d\n",
                          stats[i].num_tasks_processed);
    absl::StrAppendFormat(
        &s,
        "\t\tTime the current task spent in the queue before "
        "being picked up: %s\n",
        absl::FormatDuration(stats[i].current_task_started_at -
                             stats[i].current_task_created_at));
    absl::StrAppendFormat(
        &s, "\t\tTime the current task has been running: %s\n",
        absl::FormatDuration(absl::Now() - stats[i].current_task_started_at));
  }
  return s;
}

}  // namespace reverb
}  // namespace deepmind
