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

#include "reverb/cc/table_extensions/async_base.h"

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status.h"

namespace deepmind {
namespace reverb {
namespace {

constexpr auto kTimeout = absl::Milliseconds(50);

class TestImpl : public TableExtensionAsyncBase {
 public:
  struct Config {
    using OnItem = std::function<void(const TableItem&)>;

    OnItem on_insert = [](const TableItem&) {};
    OnItem on_update = [](const TableItem&) {};
    OnItem on_sample = [](const TableItem&) {};
    OnItem on_delete = [](const TableItem&) {};
    std::function<void()> on_reset = []() {};
  };

  TestImpl(size_t buffer_size, Config config)
      : TableExtensionAsyncBase(buffer_size), config_(std::move(config)) {}

  void ApplyOnDelete(const TableItem& item) override {
    config_.on_delete(item);
  }
  void ApplyOnInsert(const TableItem& item) override {
    config_.on_insert(item);
  }
  void ApplyOnReset() override { config_.on_reset(); }
  void ApplyOnUpdate(const TableItem& item) override {
    config_.on_update(item);
  }
  void ApplyOnSample(const TableItem& item) override {
    config_.on_sample(item);
  }

  // Expose methods that only are meant to be used by Table.
  void OnInsert(absl::Mutex* mu, const TableItem& item) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) {
    TableExtensionAsyncBase::OnInsert(mu, item);
  }
  void OnDelete(absl::Mutex* mu, const TableItem& item) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) {
    TableExtensionAsyncBase::OnDelete(mu, item);
  }
  void OnReset(absl::Mutex* mu) override ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) {
    TableExtensionAsyncBase::OnReset(mu);
  }
  void OnUpdate(absl::Mutex* mu, const TableItem& item) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) {
    TableExtensionAsyncBase::OnUpdate(mu, item);
  }
  void OnSample(absl::Mutex* mu, const TableItem& item) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) {
    TableExtensionAsyncBase::OnSample(mu, item);
  }
  tensorflow::Status RegisterTable(absl::Mutex* mu, Table* table)
      ABSL_LOCKS_EXCLUDED(mu) override {
    return TableExtensionAsyncBase::RegisterTable(mu, table);
  }
  void UnregisterTable(absl::Mutex* mu, Table* table)
      ABSL_LOCKS_EXCLUDED(mu) override {
    TableExtensionAsyncBase::UnregisterTable(mu, table);
  }

 private:
  Config config_;
};

TEST(TableExtensionAsyncBaseTest, BlocksWhenBufferFull) {
  TestImpl::Config config;

  absl::Notification block;
  config.on_insert = [&block](const TableItem&) {
    block.WaitForNotification();
  };

  TestImpl extension(1, std::move(config));

  absl::Mutex mu;
  TF_ASSERT_OK(extension.RegisterTable(&mu, nullptr));

  // First insert should be fine.
  {
    absl::MutexLock lock(&mu);
    extension.OnInsert(&mu, TableItem());
  }

  // The second call should block.
  absl::Notification done;
  auto thread = internal::StartThread("OnInsert", [&] {
    absl::MutexLock lock(&mu);
    extension.OnInsert(&mu, TableItem());
    done.Notify();
  });

  EXPECT_FALSE(done.WaitForNotificationWithTimeout(kTimeout));

  // Unblocking calls should allow the second call to complete.
  block.Notify();
  EXPECT_TRUE(done.WaitForNotificationWithTimeout(kTimeout));

  extension.UnregisterTable(&mu, nullptr);
}

TEST(TableExtensionAsyncBaseTest, WaitUntilReady) {
  TestImpl::Config config;

  absl::Notification insert_block;
  absl::Notification insert_started;
  config.on_insert = [&](const TableItem&) {
    insert_started.Notify();
    insert_block.WaitForNotification();
  };

  TestImpl extension(10, std::move(config));

  absl::Mutex mu;
  TF_ASSERT_OK(extension.RegisterTable(&mu, nullptr));

  // Start the insert.
  auto insert_thread = internal::StartThread("DoInsert", [&] {
    absl::MutexLock lock(&mu);
    extension.OnInsert(&mu, TableItem());
  });

  // Start the waiting
  absl::Notification wait_done;
  auto wait_until_ready_thread = internal::StartThread("Wait", [&] {
    insert_started.WaitForNotification();
    extension.WaitUntilReady();
    wait_done.Notify();
  });

  // The waiting should not complete until the insert has been completed.
  EXPECT_FALSE(wait_done.WaitForNotificationWithTimeout(kTimeout));
  insert_block.Notify();
  EXPECT_TRUE(wait_done.WaitForNotificationWithTimeout(kTimeout));

  extension.UnregisterTable(&mu, nullptr);
}

void BM_OnInsert(benchmark::State& state) {
  TestImpl::Config config;
  config.on_insert = [&](const TableItem&) {
    absl::SleepFor(absl::Microseconds(state.range(0)));
  };

  // Note: We are using a relatively small buffer size to be able to force the
  // extension into blocked mode in the benchmarks.
  TestImpl extension(1000, std::move(config));
  TableItem item{};

  absl::Mutex mu;
  TF_ASSERT_OK(extension.RegisterTable(&mu, nullptr));

  mu.Lock();
  for (auto _ : state) {
    extension.OnInsert(&mu, item);
  }
  mu.Unlock();

  extension.UnregisterTable(&mu, nullptr);
}

BENCHMARK(BM_OnInsert)->Range(1, 10000);

}  // namespace
}  // namespace reverb
}  // namespace deepmind
