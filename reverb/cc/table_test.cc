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

#include "reverb/cc/table.h"

#include <atomic>
#include <cfloat>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdint>
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/selectors/uniform.h"
#include "reverb/cc/support/task_executor.h"
#include "reverb/cc/table_extensions/interface.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace deepmind {
namespace reverb {
namespace {

const absl::Duration kTimeout = absl::Milliseconds(250);
const absl::Duration kLongTimeout = absl::Seconds(1);

using ::deepmind::reverb::testing::Partially;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;

MATCHER_P(HasItemKey, key, "") { return arg.item.key() == key; }
MATCHER_P(HasSampledItemKey, key, "") { return arg.ref->item.key() == key; }

template <typename... Ts>
std::unique_ptr<Table> MakeTable(Ts... args) {
  auto table = absl::make_unique<Table>(args...);
  table->SetCallbackExecutor(std::make_shared<TaskExecutor>(1, "worker"));
  return table;
}

TableItem MakeItem(uint64_t key, double priority,
                   const std::vector<SequenceRange>& sequences) {
  TableItem item;

  std::vector<ChunkData> data(sequences.size());
  for (int i = 0; i < sequences.size(); i++) {
    data[i] = testing::MakeChunkData(key * 100 + i, sequences[i]);
    item.chunks.push_back(std::make_shared<ChunkStore::Chunk>(data[i]));
  }

  item.item = testing::MakePrioritizedItem(key, priority, data);

  return item;
}

TableItem MakeItem(uint64_t key, double priority) {
  return MakeItem(key, priority, {testing::MakeSequenceRange(key * 100, 0, 1)});
}

std::shared_ptr<RateLimiter> MakeLimiter(int64_t min_size) {
  return std::make_shared<RateLimiter>(1.0, min_size, -DBL_MAX, DBL_MAX);
}

std::unique_ptr<Table> MakeUniformTable(const std::string& name,
                                        int64_t max_size = 1000,
                                        int32_t max_times_sampled = 0) {
  auto table = MakeTable(name, std::make_shared<UniformSelector>(),
                         std::make_shared<FifoSelector>(), max_size,
                         max_times_sampled, MakeLimiter(1));
  return table;
}

void WaitForTableSize(Table* table, int size) {
  for (int retry = 0; retry < 100 && size != 0; retry++) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  EXPECT_EQ(table->size(), size);
}

TEST(TableTest, SetsName) {
  auto first = MakeUniformTable("first");
  auto second = MakeUniformTable("second");
  EXPECT_EQ(first->name(), "first");
  EXPECT_EQ(second->name(), "second");
}

TEST(TableTest, CopyAfterInsert) {
  auto table = MakeUniformTable("dist");
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 123)));

  auto items = table->Copy();
  ASSERT_THAT(items, SizeIs(1));
  EXPECT_THAT(
      items[0].item,
      Partially(testing::EqualsProto("key: 3 times_sampled: 0 priority: 123")));
}

TEST(TableTest, CopySubset) {
  auto table = MakeUniformTable("dist");
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 123)));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(4, 123)));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(5, 123)));
  EXPECT_THAT(table->Copy(1), SizeIs(1));
  EXPECT_THAT(table->Copy(2), SizeIs(2));
}

TEST(TableTest, InsertOrAssignOverwrites) {
  auto table = MakeUniformTable("dist");
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 123)));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 456)));

  auto items = table->Copy();
  ASSERT_THAT(items, SizeIs(1));
  EXPECT_EQ(items[0].item.priority(), 456);
}

TEST(TableTest, UpdatesAreAppliedPartially) {
  auto table = MakeUniformTable("dist");
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 123)));
  REVERB_EXPECT_OK(table->MutateItems(
      {
          testing::MakeKeyWithPriority(5, 55),
          testing::MakeKeyWithPriority(3, 456),
      },
      {}));

  auto items = table->Copy();
  ASSERT_THAT(items, SizeIs(1));
  EXPECT_EQ(items[0].item.priority(), 456);
}

TEST(TableTest, DeletesAreAppliedPartially) {
  auto table = MakeUniformTable("dist");
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 123)));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(7, 456)));
  WaitForTableSize(table.get(), 2);
  REVERB_EXPECT_OK(table->MutateItems({}, {5, 3}));
  EXPECT_THAT(table->Copy(), ElementsAre(HasItemKey(7)));
}

TEST(TableTest, SampleBlocksWhenNotEnoughItems) {
  auto table = MakeUniformTable("dist");

  absl::Notification notification;
  auto sample_thread = internal::StartThread("", [&table, &notification] {
    Table::SampledItem item;
    REVERB_EXPECT_OK(table->Sample(&item));
    notification.Notify();
  });

  EXPECT_FALSE(notification.WaitForNotificationWithTimeout(kTimeout));

  // Inserting an item should allow the call to complete.
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 123)));
  EXPECT_TRUE(notification.WaitForNotificationWithTimeout(kLongTimeout));

  sample_thread = nullptr;  // Joins the thread.
}

TEST(TableTest, SampleMatchesInsert) {
  auto table = MakeUniformTable("dist");

  Table::Item item = MakeItem(3, 123);
  REVERB_EXPECT_OK(table->InsertOrAssign(item));

  Table::SampledItem sample;
  REVERB_EXPECT_OK(table->Sample(&sample));
  item.item.set_times_sampled(1);
  sample.ref->item.clear_inserted_at();
  EXPECT_THAT(sample.ref->item, testing::EqualsProto(item.item));
  EXPECT_EQ(sample.ref->chunks, item.chunks);
  EXPECT_EQ(sample.probability, 1);
}

TEST(TableTest, SampleIncrementsSampleTimes) {
  auto table = MakeUniformTable("dist");

  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 123)));

  Table::SampledItem item;
  EXPECT_EQ(table->Copy()[0].item.times_sampled(), 0);
  REVERB_EXPECT_OK(table->Sample(&item));
  EXPECT_EQ(table->Copy()[0].item.times_sampled(), 1);
  REVERB_EXPECT_OK(table->Sample(&item));
  EXPECT_EQ(table->Copy()[0].item.times_sampled(), 2);
}

TEST(TableTest, MaxTimesSampledIsRespected) {
  auto table = MakeUniformTable("dist", 10, 2);

  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 123)));

  Table::SampledItem item;
  EXPECT_EQ(table->Copy()[0].item.times_sampled(), 0);
  REVERB_ASSERT_OK(table->Sample(&item));
  EXPECT_EQ(table->Copy()[0].item.times_sampled(), 1);
  REVERB_ASSERT_OK(table->Sample(&item));
  EXPECT_THAT(table->Copy(), IsEmpty());
}

TEST(TableTest, SampleFlexibleBatchRequireEmptyOutputVector) {
  auto table = MakeUniformTable("dist", 10, 2);

  std::vector<Table::SampledItem> items;
  items.emplace_back();

  auto status = table->SampleFlexibleBatch(&items, 1, absl::ZeroDuration());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr(
          "Table::SampleFlexibleBatch called with non-empty output vector."));
}

TEST(TableTest, SampleNotFullBatch) {
  auto table = MakeUniformTable("dist", 10, 1);

  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 123)));

  std::vector<Table::SampledItem> items;

  REVERB_ASSERT_OK(table->SampleFlexibleBatch(&items, 2));
  EXPECT_EQ(items.size(), 1);
}

TEST(TableTest, EnqueSampleRequestSetsRateLimitedIfBlocked) {
  auto table = MakeUniformTable("table");

  absl::Notification first_done;
  Table::SampledItem rate_limited_item;
  auto first_callback =
      std::make_shared<Table::SamplingCallback>([&](Table::SampleRequest* req) {
        rate_limited_item = req->samples[0];
        first_done.Notify();
      });

  table->EnqueSampleRequest(1, first_callback, kLongTimeout);

  // Wait until the worker has picked up the request and gone back to sleep
  // since it was unable to do anything.
  while (table->num_pending_async_sample_requests() ||
         !table->worker_is_sleeping()) {
    absl::SleepFor(absl::Milliseconds(1));
  }

  bool can_insert_more;
  REVERB_ASSERT_OK(table->InsertOrAssignAsync(
      MakeItem(1, 1), &can_insert_more,
      std::make_shared<Table::InsertCallback>([](uint64_t) {})));
  ASSERT_TRUE(can_insert_more);
  ASSERT_TRUE(first_done.WaitForNotificationWithTimeout(kLongTimeout));
  EXPECT_TRUE(rate_limited_item.rate_limited);

  absl::Notification second_done;
  Table::SampledItem not_rate_limited_item;
  auto second_callback =
      std::make_shared<Table::SamplingCallback>([&](Table::SampleRequest* req) {
        not_rate_limited_item = req->samples[0];
        second_done.Notify();
      });

  table->EnqueSampleRequest(1, second_callback, kLongTimeout);
  ASSERT_TRUE(second_done.WaitForNotificationWithTimeout(kLongTimeout));
  EXPECT_FALSE(not_rate_limited_item.rate_limited);
}

TEST(TableTest, InsertDeletesWhenOverflowing) {
  auto table = MakeUniformTable("dist", 10);

  for (int i = 0; i < 15; i++) {
    REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(i, 123)));
  }
  auto items = table->Copy();
  EXPECT_THAT(items, SizeIs(10));
  for (const Table::Item& item : items) {
    EXPECT_GE(item.item.key(), 5);
    EXPECT_LT(item.item.key(), 15);
  }
}

TEST(TableTest, ConcurrentCalls) {
  auto table = MakeUniformTable("dist", 1000);

  std::vector<std::unique_ptr<internal::Thread>> bundle;
  std::atomic<int> count(0);
  for (Table::Key i = 0; i < 1000; i++) {
    bundle.push_back(internal::StartThread("", [i, &table, &count] {
      REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(i, 123)));
      Table::SampledItem item;
      REVERB_EXPECT_OK(table->Sample(&item));
      REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(i, 456)));
      count++;
    }));
  }
  bundle.clear();  // Joins all threads.
  EXPECT_EQ(count, 1000);
}

TEST(TableTest, UseAsQueue) {
  Table queue(
      /*name=*/"queue",
      /*sampler=*/std::make_shared<FifoSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/10,
      /*max_times_sampled=*/1,
      std::make_shared<RateLimiter>(
          /*samples_per_insert=*/1.0,
          /*min_size_to_sample=*/1,
          /*min_diff=*/0,
          /*max_diff=*/10.0));
  for (int i = 0; i < 10; i++) {
    REVERB_EXPECT_OK(queue.InsertOrAssign(MakeItem(i, 123)));
  }

  // This should now be blocked
  absl::Notification insert;
  auto insert_thread = internal::StartThread("", [&] {
    REVERB_EXPECT_OK(queue.InsertOrAssign(MakeItem(10, 123)));
    insert.Notify();
  });

  EXPECT_FALSE(insert.WaitForNotificationWithTimeout(kTimeout));

  for (int i = 0; i < 11; i++) {
    Table::SampledItem item;
    REVERB_EXPECT_OK(queue.Sample(&item));
    EXPECT_THAT(item, HasSampledItemKey(i));
  }

  EXPECT_TRUE(insert.WaitForNotificationWithTimeout(kLongTimeout));

  insert_thread = nullptr;  // Joins the thread.

  EXPECT_EQ(queue.size(), 0);

  // Sampling should now be blocked.
  absl::Notification sample;
  auto sample_thread = internal::StartThread("", [&] {
    Table::SampledItem item;
    REVERB_EXPECT_OK(queue.Sample(&item));
    sample.Notify();
  });

  EXPECT_FALSE(sample.WaitForNotificationWithTimeout(kTimeout));

  // Inserting a new item should result in it being sampled straight away.
  REVERB_EXPECT_OK(queue.InsertOrAssign(MakeItem(100, 123)));
  EXPECT_TRUE(sample.WaitForNotificationWithTimeout(kLongTimeout));

  EXPECT_EQ(queue.size(), 0);

  sample_thread = nullptr;  // Joins the thread.
}

TEST(TableTest, ConcurrentInsertOfTheSameKey) {
  auto table = MakeTable(
      /*name=*/"dist",
      /*sampler=*/std::make_shared<UniformSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/1000,
      /*max_times_sampled=*/0,
      std::make_shared<RateLimiter>(
          /*samples_per_insert=*/1.0,
          /*min_size_to_sample=*/1,
          /*min_diff=*/-1,
          /*max_diff=*/1));

  // Insert one item to make new inserts block.
  REVERB_ASSERT_OK(table->InsertOrAssign(MakeItem(1, 123)));  // diff = 1.0

  std::vector<std::unique_ptr<internal::Thread>> bundle;

  // Try to insert the same item 10 times. All should be blocked.
  std::atomic<int> count(0);
  for (int i = 0; i < 10; i++) {
    bundle.push_back(internal::StartThread("", [&] {
      REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(10, 123)));
      count++;
    }));
  }

  EXPECT_EQ(count, 0);

  // Making a single sample should unblock one of the inserts. The other inserts
  // are now updates but they are still waiting for their right to insert.
  Table::SampledItem item;
  REVERB_EXPECT_OK(table->Sample(&item));

  // Sampling once more would unblock one of the inserts, it will then see that
  // it is now an update and not use its right to insert. Once it releases the
  // lock the same process will follow for all the remaining inserts.
  REVERB_EXPECT_OK(table->Sample(&item));

  bundle.clear();  // Joins all threads.

  EXPECT_EQ(count, 10);
  EXPECT_EQ(table->size(), 2);
}

TEST(TableTest, CloseCancelsPendingCalls) {
  auto table = MakeTable(
      /*name=*/"dist",
      /*sampler=*/std::make_shared<UniformSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/1000,
      /*max_times_sampled=*/0,
      std::make_shared<RateLimiter>(
          /*samples_per_insert=*/1.0,
          /*min_size_to_sample=*/1,
          /*min_diff=*/-1,
          /*max_diff=*/1));

  // Insert two item to make new inserts block.
  REVERB_ASSERT_OK(table->InsertOrAssign(MakeItem(1, 123)));  // diff = 1.0

  absl::Status status;
  absl::Notification notification;
  auto thread = internal::StartThread("", [&] {
    status = table->InsertOrAssign(MakeItem(10, 123));
    notification.Notify();
  });

  EXPECT_FALSE(notification.WaitForNotificationWithTimeout(kTimeout));

  table->Close();

  EXPECT_TRUE(notification.WaitForNotificationWithTimeout(kLongTimeout));
  REVERB_EXPECT_OK(status);

  thread = nullptr;  // Joins the thread.
}

TEST(TableTest, ResetResetsRateLimiter) {
  auto table = MakeTable(
      /*name=*/"dist",
      /*sampler=*/std::make_shared<UniformSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/1000,
      /*max_times_sampled=*/0,
      std::make_shared<RateLimiter>(
          /*samples_per_insert=*/1.0,
          /*min_size_to_sample=*/1,
          /*min_diff=*/-1,
          /*max_diff=*/1));

  // Insert two item to make new inserts block.
  REVERB_ASSERT_OK(table->InsertOrAssign(MakeItem(1, 123)));  // diff = 1.0

  absl::Notification notification;
  auto thread = internal::StartThread("", [&] {
    REVERB_ASSERT_OK(table->InsertOrAssign(MakeItem(10, 123)));
    notification.Notify();
  });

  EXPECT_FALSE(notification.WaitForNotificationWithTimeout(kTimeout));

  // Resetting the table should unblock new inserts.
  REVERB_ASSERT_OK(table->Reset());

  EXPECT_TRUE(notification.WaitForNotificationWithTimeout(kLongTimeout));

  thread = nullptr;  // Joins the thread.
}

TEST(TableTest, ResetClearsAllData) {
  auto table = MakeUniformTable("dist");
  REVERB_ASSERT_OK(table->InsertOrAssign(MakeItem(1, 123)));
  EXPECT_EQ(table->size(), 1);
  REVERB_ASSERT_OK(table->Reset());
  EXPECT_EQ(table->size(), 0);
}

TEST(TableTest, ResetWhileConcurrentCalls) {
  auto table = MakeUniformTable("dist");
  std::vector<std::unique_ptr<internal::Thread>> bundle;
  for (Table::Key i = 0; i < 1000; i++) {
    bundle.push_back(internal::StartThread("", [i, &table] {
      if (i % 123 == 0) {
        REVERB_EXPECT_OK(table->Reset());
      }
      REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(i, 123)));
      auto result =
          table->MutateItems({testing::MakeKeyWithPriority(i, 456)}, {i});
      if (!result.ok()) {
        EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
      }
    }));
  }
  bundle.clear();  // Joins all threads.
}

TEST(TableTest, CheckpointOrderItems) {
  auto table = MakeUniformTable("dist");

  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(1, 123)));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 125)));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(2, 124)));

  auto checkpoint = table->Checkpoint();
  EXPECT_THAT(checkpoint.items,
              ElementsAre(Partially(testing::EqualsProto("key: 1")),
                          Partially(testing::EqualsProto("key: 3")),
                          Partially(testing::EqualsProto("key: 2"))));
}

TEST(TableTest, CheckpointSanityCheck) {
  tensorflow::StructuredValue signature;
  auto* spec =
      signature.mutable_list_value()->add_values()->mutable_tensor_spec_value();
  spec->set_dtype(tensorflow::DT_FLOAT);
  tensorflow::TensorShapeProto shape;
  tensorflow::TensorShape({1, 2}).AsProto(spec->mutable_shape());

  auto table =
      MakeTable("dist", std::make_shared<UniformSelector>(),
                std::make_shared<FifoSelector>(), 10, 1,
                std::make_shared<RateLimiter>(1.0, 3, -10, 7),
                std::vector<std::shared_ptr<TableExtension>>(), signature);

  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(1, 123)));

  auto checkpoint = table->Checkpoint();

  PriorityTableCheckpoint want;
  want.set_table_name("dist");
  want.set_max_size(10);
  want.set_max_times_sampled(1);
  want.mutable_rate_limiter()->set_samples_per_insert(1.0);
  want.mutable_rate_limiter()->set_min_size_to_sample(3);
  want.mutable_rate_limiter()->set_min_diff(-10);

  EXPECT_THAT(checkpoint.checkpoint, Partially(testing::EqualsProto(R"pb(
                table_name: 'dist'
                max_size: 10
                max_times_sampled: 1
                num_deleted_episodes: 0
                rate_limiter: {
                  samples_per_insert: 1.0
                  min_size_to_sample: 3
                  min_diff: -10
                  max_diff: 7
                  sample_count: 0
                  insert_count: 1
                }
                sampler: { uniform: true }
                remover: { fifo: true }
                signature: {
                  list_value: {
                    values: {
                      tensor_spec_value: {
                        shape: {
                          dim: { size: 1 }
                          dim: { size: 2 }
                        }
                        dtype: DT_FLOAT
                      }
                    }
                  }
                }
              )pb")));

  EXPECT_THAT(checkpoint.items,
              ElementsAre(Partially(testing::EqualsProto("key: 1"))));
}

TEST(TableTest, BlocksSamplesWhenSizeToSmallDueToAutoDelete) {
  auto table = MakeTable(
      /*name=*/"dist",
      /*sampler=*/std::make_shared<FifoSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/10,
      /*max_times_sampled=*/2,
      std::make_shared<RateLimiter>(
          /*samples_per_insert=*/1.0,
          /*min_size_to_sample=*/3,
          /*min_diff=*/0,
          /*max_diff=*/5));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(1, 1)));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(2, 1)));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 1)));

  // It should be fine to sample now as the table has been reached its min size.
  Table::SampledItem sample_1;
  REVERB_EXPECT_OK(table->Sample(&sample_1));
  EXPECT_THAT(sample_1, HasSampledItemKey(1));

  // A second sample should be fine since the table is still large enough.
  Table::SampledItem sample_2;
  REVERB_EXPECT_OK(table->Sample(&sample_2));
  EXPECT_THAT(sample_2, HasSampledItemKey(1));

  // Due to max_times_sampled, the table should have one item less which should
  // block more samples from proceeding.
  absl::Notification notification;
  auto sample_thread = internal::StartThread("", [&] {
    Table::SampledItem sample;
    REVERB_EXPECT_OK(table->Sample(&sample));
    notification.Notify();
  });
  EXPECT_FALSE(notification.WaitForNotificationWithTimeout(kTimeout));

  // Inserting a new item should unblock the sampling.
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(4, 1)));
  EXPECT_TRUE(notification.WaitForNotificationWithTimeout(kLongTimeout));

  sample_thread = nullptr;  // Joins the thread.
}

TEST(TableTest, BlocksSamplesWhenSizeToSmallDueToExplicitDelete) {
  auto table = MakeTable(
      /*name=*/"dist",
      /*sampler=*/std::make_shared<FifoSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/10,
      /*max_times_sampled=*/-1,
      std::make_shared<RateLimiter>(
          /*samples_per_insert=*/1.0,
          /*min_size_to_sample=*/3,
          /*min_diff=*/0,
          /*max_diff=*/5));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(1, 1)));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(2, 1)));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 1)));

  // It should be fine to sample now as the table has been reached its min size.
  Table::SampledItem sample_1;
  REVERB_EXPECT_OK(table->Sample(&sample_1));
  EXPECT_THAT(sample_1, HasSampledItemKey(1));

  // Deleting an item will make the table too small to allow samples.
  REVERB_EXPECT_OK(table->MutateItems({}, {1}));

  absl::Notification notification;
  auto sample_thread = internal::StartThread("", [&] {
    Table::SampledItem sample;
    REVERB_EXPECT_OK(table->Sample(&sample));
    notification.Notify();
  });
  EXPECT_FALSE(notification.WaitForNotificationWithTimeout(kTimeout));

  // Inserting a new item should unblock the sampling.
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(4, 1)));
  EXPECT_TRUE(notification.WaitForNotificationWithTimeout(kLongTimeout));

  sample_thread = nullptr;  // Joins the thread.

  // And any new samples should be fine.
  Table::SampledItem sample_2;
  REVERB_EXPECT_OK(table->Sample(&sample_2));
  EXPECT_THAT(sample_2, HasSampledItemKey(2));
}

TEST(TableTest, GetExistingItem) {
  auto table = MakeUniformTable("dist");

  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(1, 1)));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(2, 1)));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 1)));

  TableItem item;
  EXPECT_TRUE(table->Get(2, &item));
  EXPECT_THAT(item, HasItemKey(2));
}

TEST(TableTest, GetMissingItem) {
  auto table = MakeUniformTable("dist");

  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(1, 1)));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 1)));

  TableItem item;
  EXPECT_FALSE(table->Get(2, &item));
}

TEST(TableTest, SampleSetsTableSize) {
  auto table = MakeUniformTable("dist");

  for (int i = 1; i <= 10; i++) {
    REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(i, 1)));
    Table::SampledItem sample;
    REVERB_EXPECT_OK(table->Sample(&sample));
    EXPECT_EQ(sample.table_size, i);
  }
}

TEST(PriorityTableDeathTest, DiesIfUnsafeAddExtensionCalledWhenNonEmpty) {
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  auto table = MakeUniformTable("dist");
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(1, 1)));
  ASSERT_DEATH(table->UnsafeAddExtension(nullptr), "");
}

TEST(TableTest, NumEpisodes) {
  auto table = MakeUniformTable("dist");

  std::vector<SequenceRange> ranges{
      testing::MakeSequenceRange(100, 0, 5),
      testing::MakeSequenceRange(100, 6, 10),
      testing::MakeSequenceRange(101, 0, 5),
  };

  // First item has a never seen episode before.
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(1, 1, {ranges[0]})));
  EXPECT_EQ(table->num_episodes(), 1);

  // Second item references the same episode.
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(2, 1, {ranges[1]})));
  EXPECT_EQ(table->num_episodes(), 1);

  // Third item has a new episode.
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 1, {ranges[2]})));
  EXPECT_EQ(table->num_episodes(), 2);

  // Removing the second item should not change the episode count as the first
  // item is still referencing the episode.
  REVERB_EXPECT_OK(table->MutateItems({}, {2}));
  EXPECT_EQ(table->num_episodes(), 2);

  // Removing the first item should now result in the episode count reduced as
  // it is the last reference to the episode.
  REVERB_EXPECT_OK(table->MutateItems({}, {1}));
  EXPECT_EQ(table->num_episodes(), 1);

  // Resetting the table should bring the count back to zero.
  REVERB_EXPECT_OK(table->Reset());
  EXPECT_EQ(table->num_episodes(), 0);
}

TEST(TableTest, NumDeletedEpisodes) {
  auto table = MakeUniformTable("dist");

  std::vector<SequenceRange> ranges{
      testing::MakeSequenceRange(100, 0, 5),
      testing::MakeSequenceRange(100, 6, 10),
      testing::MakeSequenceRange(101, 0, 5),
  };

  // Should initially be zero.
  EXPECT_EQ(table->num_deleted_episodes(), 0);

  // Manually setting the count can be done just after construction.
  table->set_num_deleted_episodes_from_checkpoint(1);
  EXPECT_EQ(table->num_deleted_episodes(), 1);

  // Add two items referencing the same episode and one item that reference a
  // second episode. This should not have any impact on the number of deleted
  // episodes.
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(1, 1, {ranges[0]})));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(2, 1, {ranges[1]})));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(3, 1, {ranges[2]})));
  EXPECT_EQ(table->num_deleted_episodes(), 1);

  // Removing one of the items that reference the episode shared by two items
  // should not impact the number of deleted episodes.
  REVERB_EXPECT_OK(table->MutateItems({}, {2}));
  EXPECT_EQ(table->num_deleted_episodes(), 1);

  // Removing the ONLY item that references the second episode should result
  // in the deleted items count being incremented.
  REVERB_EXPECT_OK(table->MutateItems({}, {3}));
  EXPECT_EQ(table->num_deleted_episodes(), 2);

  // Removing the second (and last) item referencing the first episode should
  // also result in the deleted episodes count being incremented.
  REVERB_EXPECT_OK(table->MutateItems({}, {1}));
  EXPECT_EQ(table->num_deleted_episodes(), 3);

  // Resetting the table should bring the count back to zero.
  REVERB_EXPECT_OK(table->Reset());
  EXPECT_EQ(table->num_deleted_episodes(), 0);
}

TEST(TableDeathTest, SetNumDeletedEpisodesFromCheckpointOnNonEmptyTable) {
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  auto table = MakeUniformTable("dist");
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(1, 1)));
  ASSERT_DEATH(table->set_num_deleted_episodes_from_checkpoint(1), "");
}

TEST(TableDeathTest, SetNumDeletedEpisodesFromCheckpointCalledTwice) {
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  auto table = MakeUniformTable("dist");
  table->set_num_deleted_episodes_from_checkpoint(1);
  ASSERT_DEATH(table->set_num_deleted_episodes_from_checkpoint(1), "");
}

TEST(TableTest, Info) {
  auto table = MakeTable(
      /*name=*/"dist",
      /*sampler=*/std::make_shared<UniformSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/10,
      /*max_times_sampled=*/1,
      std::make_shared<RateLimiter>(
          /*samples_per_insert=*/1.0,
          /*min_size_to_sample=*/1,
          /*min_diff=*/-1,
          /*max_diff=*/5));
  table->set_num_deleted_episodes_from_checkpoint(5);
  table->set_num_unique_samples_from_checkpoint(2);

  // Insert two items (each with different episodes).
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(1, 1)));
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(2, 1)));

  // Sample an item. This will trigger the removal of that item since
  // `max_times_sampled` is 1.
  Table::SampledItem sample;
  REVERB_EXPECT_OK(table->Sample(&sample));
  auto info = table->info();
  info.clear_table_worker_time();

  EXPECT_THAT(info, testing::EqualsProto(R"pb(
                name: 'dist'
                sampler_options { uniform: true }
                remover_options { fifo: true is_deterministic: true }
                max_size: 10
                max_times_sampled: 1
                rate_limiter_info {
                  samples_per_insert: 1
                  min_diff: -1
                  max_diff: 5
                  min_size_to_sample: 1
                  insert_stats {
                    completed: 2
                  }
                  sample_stats {
                    completed: 1
                  }
                }
                current_size: 1
                num_episodes: 1
                num_deleted_episodes: 6
                num_unique_samples: 3
              )pb"));
}

TEST(TableTest, InsertOrAssignOfItemWithoutTrajectory) {
  auto table = MakeUniformTable("dist");

  auto item = MakeItem(1, 1);
  item.item.clear_flat_trajectory();
  auto status = table->InsertOrAssign(item);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("Item trajectory must not be empty."));
}

TEST(TableTest, InsertOrAssignOfItemWithChunkMissmatch) {
  auto table = MakeUniformTable("dist");

  auto item = MakeItem(1, 1);
  item.item.mutable_flat_trajectory()
      ->mutable_columns(0)
      ->mutable_chunk_slices(0)
      ->set_chunk_key(1337);
  auto status = table->InsertOrAssign(item);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr(
          "Item chunks does not match chunks referenced in trajectory"));
}

TEST(TableTest, InsertOrAssignOfItemWithChunkLengthMissmatch) {
  auto table = MakeUniformTable("dist");

  auto item = MakeItem(1, 1);
  item.chunks.push_back(item.chunks.front());
  auto status = table->InsertOrAssign(item);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr("The number of chunks (2) does not equal the number "
                           "of chunks referenced in item's trajectory (1)."));
}

TEST(TableTest, InsertOrAssignCanTimeout) {
  auto table = MakeTable(
      /*name=*/"table",
      /*sampler=*/std::make_shared<UniformSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/5,
      /*max_times_sampled=*/1,
      /*rate_limiter=*/
      std::make_shared<RateLimiter>(
          /*samples_per_insert=*/1.0,
          /*min_size_to_sample=*/1,
          /*min_diff=*/-1,
          /*max_diff=*/1));

  // The first item should be inserted without blockage.
  REVERB_EXPECT_OK(table->InsertOrAssign(MakeItem(1, 1)));

  // The second item should not be allowed without first sampling and thus the
  // call should timeout.
  EXPECT_EQ(
      table->InsertOrAssign(MakeItem(2, 1), absl::Milliseconds(50)).code(),
      absl::StatusCode::kDeadlineExceeded);
}

TEST(TableTest, CloseWithWorker) {
  absl::Notification notification;
  auto callback = std::make_shared<Table::SamplingCallback>(
      [&](Table::SampleRequest* sample) {
        EXPECT_FALSE(sample->status.ok());
        notification.Notify();
      });
  auto table = MakeUniformTable("table");
  table->EnqueSampleRequest(100, callback);
  // Wait until the worker has picked up the request and gone back to sleep
  // since it was unable to do anything.
  while (table->num_pending_async_sample_requests() ||
         !table->worker_is_sleeping()) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  table->Close();
  notification.WaitForNotification();
}

TEST(TableTest, SampleFromClosedTable) {
  absl::Notification notification;
  absl::Mutex mu;
  auto callback = std::make_shared<Table::SamplingCallback>(
      [&](Table::SampleRequest* sample) {
        absl::MutexLock lock(&mu);
        EXPECT_EQ(sample->status.code(), absl::StatusCode::kCancelled);
        EXPECT_THAT(
            std::string(sample->status.message()),
            ::testing::HasSubstr(
                "EnqueSampleRequest: RateLimiter has been cancelled"));
        notification.Notify();
      });
  auto table = MakeUniformTable("table");
  table->Close();
  {
    // Make sure that callback is not executed on the same thread to avoid
    // hangs.
    absl::MutexLock lock(&mu);
    table->EnqueSampleRequest(100, callback);
  }
  notification.WaitForNotification();
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
