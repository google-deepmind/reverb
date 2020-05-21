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

#include "reverb/cc/selectors/uniform.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace deepmind {
namespace reverb {
namespace {

TEST(UniformSelectorTest, ReturnValueSantiyChecks) {
  UniformSelector uniform;

  // Non existent keys cannot be deleted or updated.
  EXPECT_EQ(uniform.Delete(123).code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_EQ(uniform.Update(123, 4).code(), tensorflow::error::INVALID_ARGUMENT);

  // Keys cannot be inserted twice.
  TF_EXPECT_OK(uniform.Insert(123, 4));
  EXPECT_EQ(uniform.Insert(123, 4).code(), tensorflow::error::INVALID_ARGUMENT);

  // Existing keys can be updated and sampled.
  TF_EXPECT_OK(uniform.Update(123, 5));
  EXPECT_EQ(uniform.Sample().key, 123);

  // Existing keys cannot be deleted twice.
  TF_EXPECT_OK(uniform.Delete(123));
  EXPECT_EQ(uniform.Delete(123).code(), tensorflow::error::INVALID_ARGUMENT);
}

TEST(UniformSelectorTest, MatchesUniformSelector) {
  const int64_t kItems = 100;
  const int64_t kSamples = 1000000;
  double expected_probability = 1. / static_cast<double>(kItems);

  UniformSelector uniform;
  for (int i = 0; i < kItems; i++) {
    TF_EXPECT_OK(uniform.Insert(i, 0));
  }
  std::vector<int64_t> counts(kItems);
  for (int i = 0; i < kSamples; i++) {
    ItemSelectorInterface::KeyWithProbability sample = uniform.Sample();
    EXPECT_EQ(sample.probability, expected_probability);
    counts[sample.key]++;
  }
  for (int64_t count : counts) {
    EXPECT_NEAR(static_cast<double>(count) / static_cast<double>(kSamples),
                expected_probability, 0.05);
  }
}

TEST(UniformSelectorTest, Options) {
  UniformSelector uniform;
  EXPECT_THAT(uniform.options(), testing::EqualsProto("uniform: true"));
}

TEST(UniformDeathTest, ClearThenSample) {
  UniformSelector uniform;
  for (int i = 0; i < 100; i++) {
    TF_EXPECT_OK(uniform.Insert(i, i));
  }
  uniform.Sample();
  uniform.Clear();
  EXPECT_DEATH(uniform.Sample(), "");
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
