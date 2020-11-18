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

#include "reverb/cc/selectors/prioritized.h"

#include <cmath>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace deepmind {
namespace reverb {
namespace {

const double kInitialPriorityExponent = 1;

TEST(PrioritizedSelectorTest, ReturnValueSantiyChecks) {
  PrioritizedSelector prioritized(kInitialPriorityExponent);

  // Non existent keys cannot be deleted or updated.
  EXPECT_EQ(prioritized.Delete(123).code(),
            tensorflow::error::INVALID_ARGUMENT);
  EXPECT_EQ(prioritized.Update(123, 4).code(),
            tensorflow::error::INVALID_ARGUMENT);

  // Keys cannot be inserted twice.
  TF_EXPECT_OK(prioritized.Insert(123, 4));
  EXPECT_EQ(prioritized.Insert(123, 4).code(),
            tensorflow::error::INVALID_ARGUMENT);

  // Existing keys can be updated and sampled.
  TF_EXPECT_OK(prioritized.Update(123, 5));
  EXPECT_EQ(prioritized.Sample().key, 123);

  // Negative priorities are not allowed.
  EXPECT_EQ(prioritized.Update(123, -1).code(),
            tensorflow::error::INVALID_ARGUMENT);
  EXPECT_EQ(prioritized.Insert(456, -1).code(),
            tensorflow::error::INVALID_ARGUMENT);

  // NAN priorites are not allowed
  EXPECT_EQ(prioritized.Update(123, NAN).code(),
            tensorflow::error::INVALID_ARGUMENT);
  EXPECT_EQ(prioritized.Insert(456, NAN).code(),
            tensorflow::error::INVALID_ARGUMENT);

  // Existing keys cannot be deleted twice.
  TF_EXPECT_OK(prioritized.Delete(123));
  EXPECT_EQ(prioritized.Delete(123).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST(PrioritizedSelectorTest, AllZeroPrioritiesResultsInUniformSampling) {
  int64_t kItems = 100;
  int64_t kSamples = 1000000;
  double expected_probability = 1. / static_cast<double>(kItems);

  PrioritizedSelector prioritized(kInitialPriorityExponent);
  for (int i = 0; i < kItems; i++) {
    TF_EXPECT_OK(prioritized.Insert(i, 0));
  }
  std::vector<int64_t> counts(kItems);
  for (int i = 0; i < kSamples; i++) {
    ItemSelectorInterface::KeyWithProbability sample = prioritized.Sample();
    EXPECT_EQ(sample.probability, expected_probability);
    counts[sample.key]++;
  }
  for (int64_t count : counts) {
    EXPECT_NEAR(static_cast<double>(count) / static_cast<double>(kSamples),
                expected_probability, 0.05);
  }
}

TEST(PrioritizedSelectorTest, SampledDistributionMatchesProbabilities) {
  const int kStart = 10;
  const int kEnd = 100;
  const int kSamples = 1000000;

  PrioritizedSelector prioritized(kInitialPriorityExponent);
  double sum = 0;
  absl::BitGen bit_gen_;
  for (int i = 0; i < kEnd; i++) {
    if (absl::Uniform<double>(bit_gen_, 0, 1) < 0.5) {
      TF_EXPECT_OK(prioritized.Insert(i, i));
    } else {
      TF_EXPECT_OK(prioritized.Insert(i, 123));
      TF_EXPECT_OK(prioritized.Update(i, i));
    }
    sum += i;
  }
  // Remove the first few items.
  for (int i = 0; i < kStart; i++) {
    TF_EXPECT_OK(prioritized.Delete(i));
    sum -= i;
  }
  // Update the priorities.
  std::vector<int64_t> counts(kEnd);
  internal::flat_hash_map<ItemSelectorInterface::Key, int64_t> probabilities;
  for (int i = 0; i < kSamples; i++) {
    ItemSelectorInterface::KeyWithProbability sample = prioritized.Sample();
    probabilities[sample.key] = sample.probability;
    counts[sample.key]++;
    EXPECT_NEAR(sample.probability, sample.key / sum, 0.001);
  }
  for (int k = 0; k < kStart; k++) EXPECT_EQ(counts[k], 0);
  for (int k = kStart; k < kEnd; k++) {
    EXPECT_NEAR(static_cast<double>(counts[k]) / static_cast<double>(kSamples),
                probabilities[k], 0.05);
  }
}

TEST(PrioritizedSelectorTest, SetsPriorityExponentInOptions) {
  PrioritizedSelector prioritized_a(0.1);
  PrioritizedSelector prioritized_b(0.5);
  EXPECT_THAT(
      prioritized_a.options(),
      testing::EqualsProto(
          "prioritized: { priority_exponent: 0.1 } is_deterministic: false"));
  EXPECT_THAT(
      prioritized_b.options(),
      testing::EqualsProto(
          "prioritized: { priority_exponent: 0.5 } is_deterministic: false"));
}

TEST(PrioritizedSelector, RoundingErrors) {
  PrioritizedSelector prioritized(1.0);

  TF_EXPECT_OK(prioritized.Insert(0, 1e-15));
  for (int i = 0; i < 10000; ++i) {
    TF_EXPECT_OK(prioritized.Insert(i + 1, 0.3));
  }

  for (int i = 0; i < 10000; ++i) {
    TF_EXPECT_OK(prioritized.Delete(i + 1));
  }

  // The root node should now have a value of 1e-15. However, due to rounding
  // errors the value will be negative unless we re-initialize the tree.
  EXPECT_GE(prioritized.NodeSumTestingOnly(0), 0.0);
}

TEST(PrioritizedDeathTest, ClearThenSample) {
  PrioritizedSelector prioritized(kInitialPriorityExponent);
  for (int i = 0; i < 100; i++) {
    TF_EXPECT_OK(prioritized.Insert(i, i));
  }
  prioritized.Sample();
  prioritized.Clear();
  EXPECT_DEATH(prioritized.Sample(), "");
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
