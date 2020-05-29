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

#include "reverb/cc/selectors/heap.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/interface.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace deepmind {
namespace reverb {
namespace {

TEST(HeapSelectorTest, ReturnValueSantiyChecks) {
  HeapSelector heap;

  // Non existent keys cannot be deleted or updated.
  EXPECT_EQ(heap.Delete(123).code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_EQ(heap.Update(123, 4).code(), tensorflow::error::INVALID_ARGUMENT);

  // Keys cannot be inserted twice.
  TF_EXPECT_OK(heap.Insert(123, 4));
  EXPECT_EQ(heap.Insert(123, 4).code(), tensorflow::error::INVALID_ARGUMENT);

  // Existing keys can be updated and sampled.
  TF_EXPECT_OK(heap.Update(123, 5));
  EXPECT_EQ(heap.Sample().key, 123);

  // Existing keys cannot be deleted twice.
  TF_EXPECT_OK(heap.Delete(123));
  EXPECT_EQ(heap.Delete(123).code(), tensorflow::error::INVALID_ARGUMENT);
}

TEST(HeapSelectorTest, SampleMinPriorityFirstByDefault) {
  HeapSelector heap;

  TF_EXPECT_OK(heap.Insert(123, 2));
  TF_EXPECT_OK(heap.Insert(124, 1));
  TF_EXPECT_OK(heap.Insert(125, 3));

  EXPECT_EQ(heap.Sample().key, 124);

  // Remove the top item.
  TF_EXPECT_OK(heap.Delete(124));

  // Second lowest priority is now the lowest.
  EXPECT_EQ(heap.Sample().key, 123);
}

TEST(HeapSelectorTest, BreakTiesByInsertionOrder) {
  HeapSelector heap;

  // We insert keys with priorities such that the keys are removed in the
  // order [0, 1, 2, 3, 4, 5].
  // Three-way tie and two-way tie are checked.
  TF_EXPECT_OK(heap.Insert(5, 300));
  TF_EXPECT_OK(heap.Insert(0, 1));
  TF_EXPECT_OK(heap.Insert(3, 20));
  TF_EXPECT_OK(heap.Insert(1, 1));
  TF_EXPECT_OK(heap.Insert(4, 20));
  TF_EXPECT_OK(heap.Insert(2, 1));

  for (auto i = 0; i < 6; i++) {
    EXPECT_EQ(heap.Sample().key, i);
    TF_EXPECT_OK(heap.Delete(i));
  }
}

TEST(HeapSelectorTest, BreakTiesByUpdateOrder) {
  HeapSelector heap;

  TF_EXPECT_OK(heap.Insert(2, 1));
  TF_EXPECT_OK(heap.Insert(0, 1));
  TF_EXPECT_OK(heap.Insert(1, 1));

  // Removing keys at this point would result in the order [2, 0, 1]
  // by LRU because the priorites are equal.
  // This update does not change the priority, but does increase the update
  // recency, resulting in the new order [0, 1, 2] which we verify.
  TF_EXPECT_OK(heap.Update(2, 1));
  for (auto i = 0; i < 3; i++) {
    EXPECT_EQ(heap.Sample().key, i);
    TF_EXPECT_OK(heap.Delete(i));
  }
}

TEST(HeapSelectorTest, SampleMaxPriorityWhenMinHeapFalse) {
  HeapSelector heap(false);

  TF_EXPECT_OK(heap.Insert(123, 2));
  TF_EXPECT_OK(heap.Insert(124, 1));
  TF_EXPECT_OK(heap.Insert(125, 3));

  EXPECT_EQ(heap.Sample().key, 125);

  // Remove the top item.
  TF_EXPECT_OK(heap.Delete(125));

  // Second lowest priority is now the highest.
  EXPECT_EQ(heap.Sample().key, 123);
}

TEST(HeapSelectorTest, UpdateChangesOrder) {
  HeapSelector heap;

  TF_EXPECT_OK(heap.Insert(123, 2));
  TF_EXPECT_OK(heap.Insert(124, 1));
  TF_EXPECT_OK(heap.Insert(125, 3));

  EXPECT_EQ(heap.Sample().key, 124);

  // Update the current top item.
  TF_EXPECT_OK(heap.Update(124, 5));
  EXPECT_EQ(heap.Sample().key, 123);

  // Update another item and check that it is moved to the top.
  TF_EXPECT_OK(heap.Update(125, 0.5));
  EXPECT_EQ(heap.Sample().key, 125);
}

TEST(HeapSelectorTest, Clear) {
  HeapSelector heap;

  TF_EXPECT_OK(heap.Insert(123, 2));
  TF_EXPECT_OK(heap.Insert(124, 1));

  EXPECT_EQ(heap.Sample().key, 124);

  // Clear distibution and insert an item that should otherwise be at the end.
  heap.Clear();
  TF_EXPECT_OK(heap.Insert(125, 10));
  EXPECT_EQ(heap.Sample().key, 125);
}

TEST(HeapSelectorTest, ProbabilityIsAlwaysOne) {
  HeapSelector heap;

  for (int i = 100; i < 150; i++) {
    TF_EXPECT_OK(heap.Insert(i, i));
  }

  for (int i = 0; i < 50; i++) {
    auto sample = heap.Sample();
    EXPECT_EQ(sample.probability, 1);
    TF_EXPECT_OK(heap.Delete(sample.key));
  }
}

TEST(HeapSelectorTest, Options) {
  HeapSelector min_heap;
  HeapSelector max_heap(false);
  EXPECT_THAT(
      min_heap.options(),
      testing::EqualsProto("heap: { min_heap: true} is_deterministic: true"));
  EXPECT_THAT(
      max_heap.options(),
      testing::EqualsProto("heap: { min_heap: false } is_deterministic: true"));
}

TEST(HeapSelectorDeathTest, SampleFromEmptySelector) {
  HeapSelector heap;
  EXPECT_DEATH(heap.Sample(), "");

  TF_EXPECT_OK(heap.Insert(123, 2));
  heap.Sample();

  TF_EXPECT_OK(heap.Delete(123));
  EXPECT_DEATH(heap.Sample(), "");
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
