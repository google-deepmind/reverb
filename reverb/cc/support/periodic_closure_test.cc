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

#include "reverb/cc/support/periodic_closure.h"

#include <atomic>

#include "gtest/gtest.h"
#include "absl/time/clock.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/testing/time_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace deepmind {
namespace reverb {
namespace internal {
namespace {

void IncrementAndSleep(std::atomic_int* value, absl::Duration interval) {
  *value += 1;
  absl::SleepFor(interval);
}

TEST(PeriodicClosureTest, ObeyInterval) {
  const absl::Duration kPeriod = absl::Milliseconds(10);
  const int kCalls = 10;
  const absl::Duration timeout = (kPeriod * kCalls);

  std::atomic_int actual_calls(0);
  auto callback = [&] { IncrementAndSleep(&actual_calls, kPeriod); };

  PeriodicClosure pc(callback, kPeriod);

  REVERB_EXPECT_OK(pc.Start());
  absl::SleepFor(timeout);
  REVERB_EXPECT_OK(pc.Stop());

  // The closure could get called up to kCalls+1 times: once at time 0, once
  // at time kPeriod, once at time kPeriod*2, up to once at time
  // kPeriod*kCalls.  It could be called fewer times if, say, the machine is
  // overloaded, so let's check that:
  //   1 <= actual_calls <= (kCalls + 1).
  ASSERT_LE(1, actual_calls);
  ASSERT_LE(actual_calls, kCalls + 1);
}

// If this test hangs forever, its probably a deadlock caused by setting the
// PeriodicClosure's interval to 0ms.
TEST(PeriodicClosureTest, MinInterval) {
  const absl::Duration kCallDuration = absl::Milliseconds(10);

  std::atomic_int actual_calls(0);
  auto callback = [&] { IncrementAndSleep(&actual_calls, kCallDuration); };

  PeriodicClosure pc(callback, absl::ZeroDuration());

  REVERB_EXPECT_OK(pc.Start());

  test::WaitFor([&]() { return actual_calls > 0 && actual_calls < 3; },
                kCallDuration, 100);

  REVERB_EXPECT_OK(pc.Stop());  // we should be able to Stop()

  ASSERT_GT(actual_calls, 0);
  ASSERT_LT(actual_calls, 3);
}

std::function<void()> DoNothing() {
  return []() {};
}

TEST(PeriodicClosureDeathTest, BadInterval) {
  EXPECT_DEATH(PeriodicClosure pc(DoNothing, absl::Milliseconds(-1)),
               ".* should be >= 0");
}

TEST(PeriodicClosureDeathTest, NotStopped) {
  PeriodicClosure* pc =
      new PeriodicClosure(DoNothing(), absl::Milliseconds(10));

  REVERB_EXPECT_OK(pc->Start());
  ASSERT_DEATH(delete pc, ".* before destructed");

  REVERB_EXPECT_OK(pc->Stop());
  delete pc;
}

TEST(PeriodicClosureDeathTest, DoubleStart) {
  PeriodicClosure pc(DoNothing, absl::Milliseconds(10));

  REVERB_EXPECT_OK(pc.Start());
  auto status = pc.Start();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);

  REVERB_EXPECT_OK(pc.Stop());
}

TEST(PeriodicClosureDeathTest, DoubleStop) {
  PeriodicClosure pc(DoNothing, absl::Milliseconds(10));

  REVERB_EXPECT_OK(pc.Start());

  REVERB_EXPECT_OK(pc.Stop());
  auto status = pc.Stop();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(PeriodicClosureDeathTest, StartAfterStop) {
  PeriodicClosure pc(DoNothing, absl::Milliseconds(10));

  REVERB_EXPECT_OK(pc.Stop());
  auto status = pc.Start();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace internal
}  // namespace reverb
}  // namespace deepmind
