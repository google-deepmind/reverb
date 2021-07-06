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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace deepmind {
namespace reverb {
namespace {

ThreadStats EmptyStats() { return ThreadStats{}; }

ThreadStats IdleThread(absl::Time time) {
  ThreadStats ts;
  ts.current_task_id = 1;
  ts.current_task_created_at = time;
  ts.current_task_started_at = time;
  ts.num_tasks_processed = 2;
  return ts;
}

ThreadStats ActiveThread(absl::Time time) {
  ThreadStats ts;
  ts.current_task_id = 1;
  ts.current_task_created_at = time;
  ts.current_task_started_at = time;
  ts.num_tasks_processed = 1;
  return ts;
}

TEST(ThreadStats, LastThreadIdWhenOneTasksDidNotStart) {
  std::vector<ThreadStats> stats;
  stats.push_back(ActiveThread(absl::Now()));
  stats.push_back(EmptyStats());
  EXPECT_EQ(LastThreadId(stats), -1);
}

TEST(ThreadStats, LastThreadIdWhenThreadsAreIdle) {
  std::vector<ThreadStats> stats;
  stats.push_back(ActiveThread(absl::Now()));
  stats.push_back(IdleThread(absl::Now()));
  EXPECT_EQ(LastThreadId(stats), -1);
}

TEST(ThreadStats, LastThreadIdReturnsCorrectId) {
  std::vector<ThreadStats> stats;
  stats.push_back(ActiveThread(absl::Now()));
  stats.push_back(ActiveThread(absl::Now() + absl::Seconds(20)));
  stats.push_back(ActiveThread(absl::Now() + absl::Seconds(60)));
  EXPECT_EQ(LastThreadId(stats), 2);
}

}  // namespace

}  // namespace reverb
}  // namespace deepmind
