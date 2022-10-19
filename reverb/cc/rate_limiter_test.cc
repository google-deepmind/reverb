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

#include "reverb/cc/rate_limiter.h"

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/selectors/uniform.h"
#include "reverb/cc/table.h"
#include "reverb/cc/testing/proto_test_util.h"

namespace deepmind {
namespace reverb {

namespace {

using ::deepmind::reverb::testing::EqualsProto;
using ::deepmind::reverb::testing::Partially;

constexpr absl::Duration kTimeout = absl::Milliseconds(100);

std::unique_ptr<Table> MakeTable(const std::string &name,
                                 std::shared_ptr<RateLimiter> limiter) {
  return std::make_unique<Table>(name, std::make_unique<UniformSelector>(),
                                 std::make_unique<UniformSelector>(), 10000, 0,
                                 std::move(limiter));
}

TEST(RateLimiterTest, BlocksSamplesUntilMinInsertsReached) {
  auto limiter =
      std::make_shared<RateLimiter>(/*samples_per_insert=*/1.0,
                                    /*min_size_to_sample=*/2, /*min_diff=*/-1.0,
                                    /*max_diff=*/1.0);
  auto table = MakeTable("table", limiter);
  absl::Mutex mu;
  absl::WriterMutexLock lock(&mu);
  EXPECT_FALSE(limiter->MaybeCommitSample(&mu));
  // 1 insert is not enough so the sample should still be blocked.
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);
  EXPECT_FALSE(limiter->MaybeCommitSample(&mu));

  // 2 inserts is enough, the sampling should now be unblocked.
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);
  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));
}

TEST(RateLimiterTest, OperationsWithinTheBufferAreNotBlocked) {
  auto limiter =
      std::make_shared<RateLimiter>(/*samples_per_insert=*/1.5,
                                    /*min_size_to_sample=*/1, /*min_diff=*/-3.0,
                                    /*max_diff=*/3.1);
  auto table = MakeTable("table", limiter);
  absl::Mutex mu;

  // First insert is always fine because min_size_to_sample is not yet
  // reached. The "diff" is now 1.5.
  absl::WriterMutexLock lock(&mu);
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);

  // Second insert should be fine as the "diff" after the insert is 3.0 which
  // is part of the buffer range.
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);

  // Sample calls should not be blocked as long as diff is >= -3.0.
  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));  // diff = 2.0.
  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));  // diff = 1.0.
  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));  // diff = 0.0.
  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));  // diff = -1.0.
  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));  // diff = -2.0.
  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));  // diff = -3.0.
}

TEST(RateLimiterTest, BlocksCallsThatExceedsTheMinMaxLimits) {
  auto limiter =
      std::make_shared<RateLimiter>(/*samples_per_insert=*/1.5,
                                    /*min_size_to_sample=*/2, /*min_diff=*/-1.0,
                                    /*max_diff=*/3.0);
  auto table = MakeTable("table", limiter);
  absl::Mutex mu;
  absl::WriterMutexLock lock(&mu);
  EXPECT_FALSE(limiter->MaybeCommitSample(&mu));

  // 1 insert is not enough so the sample should still be blocked.
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));  // diff = 1.5
  limiter->Insert(&mu);

  EXPECT_FALSE(limiter->MaybeCommitSample(&mu));

  // 2 inserts is enough, the sampling should now be unblocked.
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);  // diff = 3.0

  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));

  // Inserts should now be blocked as it should lead to diff = 3.5.
  EXPECT_FALSE(limiter->CanInsert(&mu, 1));

  // But adding a new sample should allow it to proceed.
  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));

  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);
}

TEST(RateLimiterTest, CanSample) {
  auto limiter =
      std::make_shared<RateLimiter>(/*samples_per_insert=*/1.0,
                                    /*min_size_to_sample=*/1, /*min_diff=*/-1.0,
                                    /*max_diff=*/1.0);
  auto table = MakeTable("table", limiter);
  absl::Mutex mu;
  absl::WriterMutexLock lock(&mu);

  // Min size should not have been reached so no samples should be allowed.
  EXPECT_FALSE(limiter->CanSample(&mu, 1));

  // Insert a single item.
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);

  // It should now be possible to sample at most two items.
  EXPECT_TRUE(limiter->CanSample(&mu, 1));   // diff = 0.
  EXPECT_TRUE(limiter->CanSample(&mu, 2));   // diff = -1.0.
  EXPECT_FALSE(limiter->CanSample(&mu, 3));  // diff = -2.0.
}

TEST(RateLimiterTest, MaybeCommitSample) {
  auto limiter =
      std::make_shared<RateLimiter>(/*samples_per_insert=*/1.0,
                                    /*min_size_to_sample=*/1, /*min_diff=*/-1.0,
                                    /*max_diff=*/1.0);
  auto table = MakeTable("table", limiter);
  absl::Mutex mu;
  absl::WriterMutexLock lock(&mu);

  // Min size should not have been reached so no samples should be allowed.
  EXPECT_FALSE(limiter->MaybeCommitSample(&mu));

  // Insert a single item.
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);

  // It should now be possible to sample at most two items.
  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));   // diff = 0.
  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));   // diff = -1.0.
  EXPECT_FALSE(limiter->MaybeCommitSample(&mu));  // diff = -2.0.
}

TEST(RateLimiterTest, CanInsert) {
  auto limiter =
      std::make_shared<RateLimiter>(/*samples_per_insert=*/1.5,
                                    /*min_size_to_sample=*/2, /*min_diff=*/0.0,
                                    /*max_diff=*/5.0);
  auto table = MakeTable("table", limiter);
  absl::Mutex mu;
  absl::WriterMutexLock lock(&mu);

  // The min size allows for the first two inserts and the error buffer allows
  // for one additional insert.
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));   // diff = 1.5 (lt min size).
  EXPECT_TRUE(limiter->CanInsert(&mu, 2));   // diff = 3.0 (eq min size).
  EXPECT_TRUE(limiter->CanInsert(&mu, 3));   // diff = 4.5.
  EXPECT_FALSE(limiter->CanInsert(&mu, 4));  // diff = 6.0

  // Do the inserts.
  for (int i = 0; i < 3; i++) {
    EXPECT_TRUE(limiter->CanInsert(&mu, 1));
    limiter->Insert(&mu);
  }

  // No inserts should be allowed now.
  EXPECT_FALSE(limiter->CanInsert(&mu, 1));  // diff = 6.0

  // Move the cursor by sampling two items.
  EXPECT_TRUE(
      limiter->MaybeCommitSample(&mu));  // diff = 3.5
  EXPECT_TRUE(
      limiter->MaybeCommitSample(&mu));  // diff = 2.5

  // One more sample should now be allowed.
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));   // diff = 4.0.
  EXPECT_FALSE(limiter->CanInsert(&mu, 2));  // diff = 5.5.
}

TEST(RateLimiterTest, CheckpointSetsBasicOptions) {
  auto limiter =
      std::make_shared<RateLimiter>(/*samples_per_insert=*/1.5,
                                    /*min_size_to_sample=*/2, /*min_diff=*/0.0,
                                    /*max_diff=*/5.0);
  auto table = MakeTable("table", limiter);
  absl::Mutex mu;
  absl::WriterMutexLock lock(&mu);
  EXPECT_THAT(limiter->CheckpointReader(&mu),
              testing::EqualsProto("samples_per_insert: 1.5 min_diff: 0 "
                                   "max_diff: 5 min_size_to_sample: 2"));
}

TEST(RateLimiterTest, CheckpointSetsInsertAndDeleteAndSampleCount) {
  auto limiter =
      std::make_shared<RateLimiter>(/*samples_per_insert=*/1.5,
                                    /*min_size_to_sample=*/2, /*min_diff=*/0.0,
                                    /*max_diff=*/5.0);
  auto table = MakeTable("table", limiter);
  absl::Mutex mu;
  absl::WriterMutexLock lock(&mu);

  EXPECT_THAT(
      limiter->CheckpointReader(&mu),
      Partially(testing::EqualsProto("sample_count: 0 insert_count: 0")));

  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);
  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));
  limiter->Delete(&mu);

  EXPECT_THAT(limiter->CheckpointReader(&mu),
              Partially(testing::EqualsProto(
                  "sample_count: 1 insert_count: 2 delete_count: 1")));
}

TEST(RateLimiterTest, CanBeRestoredFromCheckpoint) {
  auto limiter =
      std::make_shared<RateLimiter>(/*samples_per_insert=*/1.5,
                                    /*min_size_to_sample=*/2, /*min_diff=*/0.0,
                                    /*max_diff=*/5.0);
  auto table = MakeTable("table", limiter);
  absl::Mutex mu;
  absl::WriterMutexLock lock(&mu);

  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);
  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));
  limiter->Delete(&mu);

  // Create a checkpoint and check its content.
  auto checkpoint = limiter->CheckpointReader(&mu);
  EXPECT_THAT(checkpoint, testing::EqualsProto("samples_per_insert: 1.5 "
                                               "min_diff: 0 "
                                               "max_diff: 5 "
                                               "min_size_to_sample: 2 "
                                               "sample_count: 1 "
                                               "insert_count: 2 "
                                               "delete_count: 1"));

  // Create a new RateLimiter from the checkpoint and verify that it behaves as
  // expected and that checkpoints generated from the restored RateLimiter
  // includes both new and inherited information.
  auto restored = std::make_shared<RateLimiter>(checkpoint);
  table = MakeTable("table", restored);

  EXPECT_TRUE(restored->CanInsert(&mu, 1));
  restored->Insert(&mu);
  EXPECT_TRUE(restored->MaybeCommitSample(&mu));

  EXPECT_THAT(restored->CheckpointReader(&mu),
              testing::EqualsProto("samples_per_insert: 1.5 "
                                   "min_diff: 0 "
                                   "max_diff: 5 "
                                   "min_size_to_sample: 2 "
                                   "sample_count: 2 "
                                   "insert_count: 3 "
                                   "delete_count: 1"));
}

TEST(RateLimiterTest, UnblocksInsertsIfDeletedItemsBringsSizeBelowMinSize) {
  auto limiter =
      std::make_shared<RateLimiter>(/*samples_per_insert=*/5,
                                    /*min_size_to_sample=*/2, /*min_diff=*/0.0,
                                    /*max_diff=*/5.0);
  auto table = MakeTable("table", limiter);
  absl::Mutex mu;
  absl::WriterMutexLock lock(&mu);
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);

  // No more inserts should be allowed until now.
  EXPECT_FALSE(limiter->CanInsert(&mu, 1));

  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));

  // The insert should still be blocked due to the large samples_per_insert.
  EXPECT_FALSE(limiter->CanInsert(&mu, 1));

  // If we remove an item then the min size is no reached which should unblock
  // the insert.
  limiter->Delete(&mu);
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
}

TEST(RateLimiterTest, BlocksSamplesIfDeleteBringsSizeBelowMinSize) {
  auto limiter =
      std::make_shared<RateLimiter>(/*samples_per_insert=*/5,
                                    /*min_size_to_sample=*/2, /*min_diff=*/0.0,
                                    /*max_diff=*/5.0);
  auto table = MakeTable("table", limiter);
  absl::Mutex mu;
    absl::WriterMutexLock lock(&mu);

  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  limiter->Insert(&mu);

  // Sampling should be fine now since the min size has been reached.
  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));

  // Deleting an item will bring the size back below the
  // min_size_to_sample which should block any further samples.
  limiter->Delete(&mu);

  EXPECT_FALSE(limiter->MaybeCommitSample(&mu));

  // Inserting a new item will bring the size up again which should unblock the
  // sampling. It should however not be unblocked by simply staging the insert.
  EXPECT_TRUE(limiter->CanInsert(&mu, 1));
  EXPECT_FALSE(limiter->MaybeCommitSample(&mu));
  limiter->Insert(&mu);
  EXPECT_TRUE(limiter->MaybeCommitSample(&mu));
}

TEST(RateLimiterTest, Info) {
  absl::Mutex mu;
  absl::ReaderMutexLock lock(&mu);

  EXPECT_THAT(RateLimiter(1, 1, 0, 5).Info(&mu),
              EqualsProto("samples_per_insert: 1 "
                          "min_size_to_sample: 1 "
                          "min_diff: 0 "
                          "max_diff: 5 "
                          "insert_stats: { "
                          "} "
                          "sample_stats: { "
                          "}"));
  EXPECT_THAT(RateLimiter(1.5, 14, -10, 5.3).Info(&mu),
              EqualsProto("samples_per_insert: 1.5 "
                          "min_size_to_sample: 14 "
                          "min_diff: -10 "
                          "max_diff: 5.3 "
                          "insert_stats: { "
                          "} "
                          "sample_stats: { "
                          "}"));
}

TEST(RateLimiterDeathTest, DiesIfMinSizeToSampleNonPositive) {
  ASSERT_DEATH(RateLimiter(1, 0, 0, 5), "");
  ASSERT_DEATH(RateLimiter(1, -1, 0, 5), "");
}

TEST(RateLimiterDeathTest, DiesIfSamplesPerInsertNonPositive) {
  ASSERT_DEATH(RateLimiter(0, 1, -100, 100), "");
  ASSERT_DEATH(RateLimiter(-1, 1, -100, 100), "");
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
