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

#include "reverb/cc/selectors/fifo.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/testing/proto_test_util.h"

namespace deepmind {
namespace reverb {
namespace {

TEST(FifoSelectorTest, ReturnValueSantiyChecks) {
  FifoSelector fifo;

  // Non existent keys cannot be deleted or updated.
  EXPECT_EQ(fifo.Delete(123).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(fifo.Update(123, 4).code(), absl::StatusCode::kInvalidArgument);

  // Keys cannot be inserted twice.
  REVERB_EXPECT_OK(fifo.Insert(123, 4));
  EXPECT_THAT(fifo.Insert(123, 4).code(), absl::StatusCode::kInvalidArgument);

  // Existing keys can be updated and sampled.
  REVERB_EXPECT_OK(fifo.Update(123, 5));
  EXPECT_EQ(fifo.Sample().key, 123);

  // Existing keys cannot be deleted twice.
  REVERB_EXPECT_OK(fifo.Delete(123));
  EXPECT_THAT(fifo.Delete(123).code(), absl::StatusCode::kInvalidArgument);
}

TEST(FifoSelectorTest, MatchesFifoOrdering) {
  int64_t kItems = 100;

  FifoSelector fifo;
  // Insert items.
  for (int i = 0; i < kItems; i++) {
    REVERB_EXPECT_OK(fifo.Insert(i, 0));
  }
  // Delete every 10th item.
  for (int i = 0; i < kItems; i++) {
    if (i % 10 == 0) {
      REVERB_EXPECT_OK(fifo.Delete(i));
    }
  }

  for (int i = 0; i < kItems; i++) {
    if (i % 10 == 0) continue;
    ItemSelector::KeyWithProbability sample = fifo.Sample();
    EXPECT_EQ(sample.key, i);
    EXPECT_EQ(sample.probability, 1);
    REVERB_EXPECT_OK(fifo.Delete(sample.key));
  }
}

TEST(FifoSelectorTest, Options) {
  FifoSelector fifo;
  EXPECT_THAT(fifo.options(),
              testing::EqualsProto("fifo: true is_deterministic: true"));
}

TEST(FifoDeathTest, ClearThenSample) {
  FifoSelector fifo;
  for (int i = 0; i < 100; i++) {
    REVERB_EXPECT_OK(fifo.Insert(i, i));
  }
  fifo.Sample();
  fifo.Clear();
  EXPECT_DEATH(fifo.Sample(), "");
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
