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

#ifndef REVERB_CC_TASK_WORKER_H_
#define REVERB_CC_TASK_WORKER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/support/periodic_closure.h"
#include "reverb/cc/support/unbounded_queue.h"
#include "reverb/cc/table.h"
#include "reverb/cc/thread_stats.h"

namespace deepmind {
namespace reverb {

struct InsertTaskInfo {
  Table::Item item;        // Item to be inserted.
  std::shared_ptr<Table> table;  // Table where the item should be inserted
  std::string DebugString() const {
    return absl::StrFormat("InsertTask{item key: %d, table: %s}", item.key(),
                           item.table());
  }
};

struct SampleTaskInfo {
  absl::Duration timeout;        // Timeout used when running the callback.
  std::shared_ptr<Table> table;  // Table to sample from.
  int32_t fetched_samples;  // Number of samples fetched before this task.
  int32_t requested_samples;  // Number of total samples requested.
  int32_t last_batch_size;  // # of items retrieved in the previous response.
  std::string DebugString() const {
    return absl::StrFormat(
        "SampleTask{table: %s, requested_samples: %d, samples_fetched_so_far: "
        "%d}",
        table->name(), requested_samples, fetched_samples);
  }
};

// Function that takes an item, a status and a boolean indicating if there is
// still enough free slots in the queue. It performs an insertion in a
// table if the status is ok. If enough_queue_slots is true, it may
// unblock the reception of new items.
typedef std::function<void(InsertTaskInfo task_info, const absl::Status&,
                           bool enough_queue_slots)>
    InsertCallback;

typedef std::function<void(SampleTaskInfo task_info, const absl::Status&,
                           bool enough_queue_slots)>
    SampleCallback;

// Class that implements a thread pool that executes tasks from a queue. It is
// thread-safe.
// In addition to the worker threads dedicated to run the tasks, a periodic
// closure checks for potential deadlocks and warns if the queue has been
// blocked for more than 1 minute.
// * TaskInfo is a class containing task information, as well as a method
//   `string DebugString() const` that returns a string with information about
//   the task (used for debugging and logging).
// * TaskCallback is a function with the following signature:
//   `void task_callback(TaskInfo task_info, const absl::Status&,
//                           bool enough_queue_slots)`
//   This function will be called with the threads of the worker.
template <class TaskInfo, class TaskCallback>
class TaskWorker {
 public:
  // Constructs an TaskWorker.
  // num_threads: number of threads that will run tasks.
  // max_queue_size_to_warn: used to indicate that the queue is growing and
  //   we should stop inserting new tasks to avoid OOMs. If the value is
  //   negative it will be ignored.
  // thread_name_prefix: is used as a prefix for the name of the threads.
  TaskWorker(size_t num_threads, size_t max_queue_size_to_warn,
             const std::string& thread_name_prefix);

  ~TaskWorker();

  // Schedules `task_cb` to be called with `item` as soon as possible. Returns
  // true if the number of items (including `item`) in the queue exceeds
  // `max_queue_size_to_warn_ (returns false otherwise).
  bool Schedule(TaskInfo task_info, TaskCallback task_cb);

  // Closes the thread pool and the queue. After calling this, no new tasks
  // will be scheduled and pending tasks will run with a cancelled status.
  void Close();

  // Returns a summary string description.
  std::string DebugString() const;

  // Returns a snapshot of the statistics of the worker threads (each item
  // of the vector corresponds to one worker thread).
  std::vector<ThreadStats> GetThreadStats() const;

  static constexpr auto kDeadlockCheckerPeriod = absl::Seconds(20);
  static constexpr auto kDeadlockCheckerTimeToWarn = absl::Minutes(1);

 private:
  struct Task {
    TaskCallback callback;
    TaskInfo task_info;
    absl::Time created_at;
  };

  struct ThreadStatsMutex {
    ThreadStats stats;
    // We use unique_ptr to make the struct movable.
    std::unique_ptr<absl::Mutex> mu = std::make_unique<absl::Mutex>();
  };

  void RunWorker(std::shared_ptr<ThreadStatsMutex> thread_stats);

  // Returns true if the number of elements in the queue is <
  // max_queue_size_to_warn_.
  bool QueueIsNotAlmostFull() const;

  // The DeadlockChecker runs periodically to check the state of the
  // TaskWorker threads. It gets the current stats, checks them, and logs if a
  // situation of deadlock is likely to be happening: if all threads have been
  // blocked by the rate limiter for at least kDeadlockCheckerTimeToWarn.
  void RunDeadlockChecker();

  internal::PeriodicClosure deadlock_checker_;

  internal::UnboundedQueue<Task> queue_;
  std::vector<std::unique_ptr<internal::Thread>> threads_;

  // When picking up a task, if it has been more than this time in the queue,
  // we will log a warning and the queue size.
  static constexpr auto kQueueTimeToWarn = absl::Seconds(10);
  std::vector<std::shared_ptr<ThreadStatsMutex>> thread_stats_;
  size_t max_queue_size_to_warn_;
};

typedef TaskWorker<InsertTaskInfo, InsertCallback> InsertWorker;
typedef TaskWorker<SampleTaskInfo, SampleCallback> SampleWorker;
/*****************************************************************************
 * Implementation of the template                                            *
 *****************************************************************************/

// Definition of the static constexpr for the linker.
template <class TaskInfo, class TaskCallback> constexpr absl::Duration
TaskWorker<TaskInfo, TaskCallback>::kDeadlockCheckerPeriod;
template <class TaskInfo, class TaskCallback> constexpr absl::Duration
TaskWorker<TaskInfo, TaskCallback>::kDeadlockCheckerTimeToWarn;

template <class TaskInfo, class TaskCallback>
TaskWorker<TaskInfo, TaskCallback>::TaskWorker(
    size_t num_threads, size_t max_queue_size_to_warn,
    const std::string& thread_name_prefix)
    : deadlock_checker_([this] { RunDeadlockChecker(); },
                        kDeadlockCheckerPeriod),
      queue_(),
      max_queue_size_to_warn_(max_queue_size_to_warn) {
  for (int thread_index = 0; thread_index < num_threads; thread_index++) {
    auto stats = std::make_shared<ThreadStatsMutex>();
    thread_stats_.push_back(stats);
    threads_.push_back(internal::StartThread(
        absl::StrCat(thread_name_prefix, "_", thread_index),
        [this, stats] { RunWorker(stats); }));
  }
  REVERB_CHECK_OK(deadlock_checker_.Start());
}

template <class TaskInfo, class TaskCallback>
TaskWorker<TaskInfo, TaskCallback>::~TaskWorker() {
  Close();
}

template <class TaskInfo, class TaskCallback>
bool TaskWorker<TaskInfo, TaskCallback>::Schedule(TaskInfo task_info,
                                                  TaskCallback task_cb) {
  queue_.Push({.callback = task_cb,
               .task_info = std::move(task_info),
               .created_at = absl::Now()});
  return QueueIsNotAlmostFull();
}

template <class TaskInfo, class TaskCallback>
void TaskWorker<TaskInfo, TaskCallback>::Close() {
  Task task;
  // Before closing, we cancel all the pending tasks.
  queue_.SetLastItemPushed();
  auto status = deadlock_checker_.Stop();
  REVERB_LOG_IF(REVERB_ERROR, !status.ok())
      << "Error when stopping the Deadlock Checker: " << status.ToString();
  while (queue_.Pop(&task)) {
    task.callback(std::move(task.task_info),
                  absl::CancelledError("Task queue is closed."),
                  /*enough_queue_slots=*/false);
  }
  threads_.clear();  // Joins worker threads.
}

template <class TaskInfo, class TaskCallback>
void TaskWorker<TaskInfo, TaskCallback>::RunWorker(
    std::shared_ptr<ThreadStatsMutex> thread_stats) {
  Task task;
  while (queue_.Pop(&task)) {
    if (auto time_in_queue = absl::Now() - task.created_at;
        time_in_queue >= kQueueTimeToWarn) {
      REVERB_LOG(REVERB_WARNING)
          << " A task spent " << absl::FormatDuration(time_in_queue)
          << " in the task queue (current queue size is: " << queue_.size()
          << "). This indicates the server is too slow in processing tasks. "
             "It could indicate that the tasks are blocked on the rate "
             "limiter.";
    }
    {
      absl::MutexLock lock(thread_stats->mu.get());
      thread_stats->stats.current_task_id++;
      thread_stats->stats.current_task_started_at = absl::Now();
      thread_stats->stats.current_task_created_at = task.created_at;
      thread_stats->stats.current_task_info = task.task_info.DebugString();
    }
    // The callback can be expensive, so we run it without holding the lock.
    task.callback(std::move(task.task_info), absl::OkStatus(),
                  QueueIsNotAlmostFull());
    {
      absl::MutexLock lock(thread_stats->mu.get());
      thread_stats->stats.num_tasks_processed++;
    }
  }
}

template <class TaskInfo, class TaskCallback>
bool TaskWorker<TaskInfo, TaskCallback>::QueueIsNotAlmostFull() const {
  if (max_queue_size_to_warn_ < 0) {
    return true;
  }
  return queue_.size() <= max_queue_size_to_warn_;
}

template <class TaskInfo, class TaskCallback>
std::vector<ThreadStats> TaskWorker<TaskInfo, TaskCallback>::GetThreadStats()
    const {
  std::vector<ThreadStats> stats(thread_stats_.size());
  for (int i = 0; i < thread_stats_.size(); i++) {
    absl::MutexLock lock(thread_stats_[i]->mu.get());
    stats[i] = thread_stats_[i]->stats;
  }
  return stats;
}

template <class TaskInfo, class TaskCallback>
void TaskWorker<TaskInfo, TaskCallback>::RunDeadlockChecker() {
  // The deadlock situation happens when all tasks have been in process for a
  // long time, this may indicate that they are all blocked by a rate limiter.
  // To track this situation, we track when the last task started and
  // which thread is working on it.
  // TODO(b/178566313) add unit tests for the deadlock checker.
  auto stats = GetThreadStats();
  int last_thread_id = LastThreadId(stats);
  if (last_thread_id == -1) {
    // There is a thread waiting for a new task, so the server is not blocked.
    return;
  }
  auto last_task_started = stats[last_thread_id].current_task_started_at;

  auto elapsed_time = absl::Now() - last_task_started;
  if (elapsed_time >= kDeadlockCheckerTimeToWarn) {
    REVERB_LOG(REVERB_WARNING)
        << " All task workers have been blocked "
        << " for at least " << absl::FormatDuration(elapsed_time)
        << "\nCurrent thread stats are:\n"
        << FormatThreadStats(stats);
  }
}

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_TASK_WORKER_H_
