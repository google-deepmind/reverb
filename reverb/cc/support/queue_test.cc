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

#include "reverb/cc/support/queue.h"

#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/synchronization/notification.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/thread.h"

namespace deepmind {
namespace reverb {
namespace internal {
namespace {

TEST(QueueTest, PushAndPopAreConsistent) {
  Queue<int> q(10);
  int output;
  for (int i = 0; i < 100; i++) {
    q.Push(i);
    q.Pop(&output);
    EXPECT_EQ(output, i);
  }
}

TEST(QueueTest, PushBlocksWhenFull) {
  Queue<int> q(2);
  ASSERT_TRUE(q.Push(1));
  ASSERT_TRUE(q.Push(2));
  absl::Notification n;
  auto t = StartThread("", [&q, &n] {
    REVERB_CHECK(q.Push(3));
    n.Notify();
  });
  ASSERT_FALSE(n.HasBeenNotified());
  int output;
  ASSERT_TRUE(q.Pop(&output));
  n.WaitForNotification();
  EXPECT_EQ(output, 1);
}

TEST(QueueTest, PopBlocksWhenEmpty) {
  Queue<int> q(2);
  absl::Notification n;
  int output;
  auto t = StartThread("", [&q, &n, &output] {
    REVERB_CHECK(q.Pop(&output));
    n.Notify();
  });
  ASSERT_FALSE(n.HasBeenNotified());
  ASSERT_TRUE(q.Push(1));
  n.WaitForNotification();
  EXPECT_EQ(output, 1);
}

TEST(QueueTest, AfterClosePushAndPopReturnFalse) {
  Queue<int> q(2);
  q.Close();
  EXPECT_FALSE(q.Push(1));
  EXPECT_FALSE(q.Pop(nullptr));
}

TEST(QueueTest, CloseUnblocksPush) {
  Queue<int> q(2);
  ASSERT_TRUE(q.Push(1));
  ASSERT_TRUE(q.Push(2));
  absl::Notification n;
  bool ok;
  auto t = StartThread("", [&q, &n, &ok] {
    ok = q.Push(3);
    n.Notify();
  });
  ASSERT_FALSE(n.HasBeenNotified());
  q.Close();
  n.WaitForNotification();
  EXPECT_FALSE(ok);
}

TEST(QueueTest, CloseUnblocksPop) {
  Queue<int> q(2);
  absl::Notification n;
  bool ok;
  auto t = StartThread("", [&q, &n, &ok] {
    int output;
    ok = q.Pop(&output);
    n.Notify();
  });
  ASSERT_FALSE(n.HasBeenNotified());
  q.Close();
  n.WaitForNotification();
  EXPECT_FALSE(ok);
}

TEST(QueueTest, SizeReturnsNumberOfElements) {
  Queue<int> q(3);
  EXPECT_EQ(q.size(), 0);

  q.Push(20);
  q.Push(30);
  EXPECT_EQ(q.size(), 2);

  int v;
  ASSERT_TRUE(q.Pop(&v));
  EXPECT_EQ(q.size(), 1);
}

}  // namespace
}  // namespace internal
}  // namespace reverb
}  // namespace deepmind
