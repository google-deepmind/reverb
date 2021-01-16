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

#include "reverb/cc/selectors/lifo.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace deepmind {
namespace reverb {
namespace {

TEST(LifoSelectorTest, ReturnValueSantiyChecks) {
  LifoSelector lifo;

  // Non existent keys cannot be deleted or updated.
  EXPECT_EQ(lifo.Delete(123).code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_EQ(lifo.Update(123, 4).code(), tensorflow::error::INVALID_ARGUMENT);

  // Keys cannot be inserted twice.
  TF_EXPECT_OK(lifo.Insert(123, 4));
  EXPECT_THAT(lifo.Insert(123, 4).code(), tensorflow::error::INVALID_ARGUMENT);

  // Existing keys can be updated and sampled.
  TF_EXPECT_OK(lifo.Update(123, 5));
  EXPECT_EQ(lifo.Sample().key, 123);

  // Existing keys cannot be deleted twice.
  TF_EXPECT_OK(lifo.Delete(123));
  EXPECT_THAT(lifo.Delete(123).code(), tensorflow::error::INVALID_ARGUMENT);
}

TEST(LifoSelectorTest, MatchesLifoOrdering) {
  int64_t kItems = 100;

  LifoSelector lifo;
  // Insert items.
  for (int i = 0; i < kItems; i++) {
    TF_EXPECT_OK(lifo.Insert(i, 0));
  }
  // Delete every 10th item.
  for (int i = 0; i < kItems; i++) {
    if (i % 10 == 0) TF_EXPECT_OK(lifo.Delete(i));
  }

  for (int i = kItems - 1; i >= 0; i--) {
    if (i % 10 == 0) continue;
    ItemSelector::KeyWithProbability sample = lifo.Sample();
    EXPECT_EQ(sample.key, i);
    EXPECT_EQ(sample.probability, 1);
    TF_EXPECT_OK(lifo.Delete(sample.key));
  }
}

TEST(LifoSelectorTest, Options) {
  LifoSelector lifo;
  EXPECT_THAT(lifo.options(),
              testing::EqualsProto("lifo: true is_deterministic: true"));
}

TEST(LifoSelectorDeathTest, ClearThenSample) {
  LifoSelector lifo;
  for (int i = 0; i < 100; i++) {
    TF_EXPECT_OK(lifo.Insert(i, i));
  }
  lifo.Sample();
  lifo.Clear();
  EXPECT_DEATH(lifo.Sample(), "");
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
