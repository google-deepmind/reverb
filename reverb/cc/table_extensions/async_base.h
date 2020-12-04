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

#ifndef REVERB_CC_TABLE_EXTENSIONS_ASYNC_BASE_H_
#define REVERB_CC_TABLE_EXTENSIONS_ASYNC_BASE_H_

#include "absl/base/thread_annotations.h"
#include "absl/container/fixed_array.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/table.h"
#include "reverb/cc/table_extensions/interface.h"

namespace deepmind {
namespace reverb {

// Async base implementation for TableExtensionInterface.
//
// This class decouples the latency sensitive parts of Table from the extension.
// Children extending this class will observe the changes, without concurrency,
// in the correct order just like they are in `TableExtensionBase`. However,
// instead of invoking the extension while holding the lock, the call is
// enqueued and eventually invoked by a single threaded worker.
//
class TableExtensionAsyncBase : public TableExtensionInterface {
 public:
  static const size_t kDefaultBufferSize = 10000;

  explicit TableExtensionAsyncBase(size_t buffer_size = kDefaultBufferSize);

  // Sanity check that extension has been unregistered from parent table.
  ~TableExtensionAsyncBase();

  // Children should override these (noop by default).
  virtual void ApplyOnDelete(const TableItem& item);
  virtual void ApplyOnInsert(const TableItem& item);
  virtual void ApplyOnReset();
  virtual void ApplyOnUpdate(const TableItem& item);
  virtual void ApplyOnSample(const TableItem& item);

  // Snapshots the current buffer state and waits until the consumer has caught
  // up. Note that that this does not mean that the buffer is empty when the
  // function returns, just that the requests pending at the start of the call
  // have been completed.
  void WaitUntilReady();

 protected:
  friend class Table;

  // Validates table, saves it to table_ and starts worker_.
  tensorflow::Status RegisterTable(absl::Mutex* mu, Table* table)
      ABSL_LOCKS_EXCLUDED(mu) override;

  // Removes table_ and stops and reemoves worker_.
  void UnregisterTable(absl::Mutex* mu, Table* table)
      ABSL_LOCKS_EXCLUDED(mu) override;

  // Delegates call to ApplyOnDelete.
  void OnDelete(absl::Mutex* mu, const TableItem& item) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Delegates call to ApplyOnInsert.
  void OnInsert(absl::Mutex* mu, const TableItem& item) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Delegates call to ApplyOnReset.
  void OnReset(absl::Mutex* mu) override ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Delegates call to ApplyOnUpdate.
  void OnUpdate(absl::Mutex* mu, const TableItem& item) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Delegates call to ApplyOnSample.
  void OnSample(absl::Mutex* mu, const TableItem& item) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

 protected:
  absl::Mutex table_mu_;
  Table* table_ ABSL_GUARDED_BY(table_mu_) = nullptr;

 private:
  void RunWorker();

  struct Request {
    enum class CallType { kDelete, kInsert, kReset, kUpdate, kSample };
    CallType call_type;
    TableItem item;
  };

  // Blocks until `next_idx_ % buffer_.size()` has been handled by the worker
  // then moves `request` into the empty slot in `buffer_`.
  void Push(Request request) ABSL_LOCK_RETURNED(idx_mu_);

  absl::Mutex idx_mu_;
  size_t next_idx_ ABSL_GUARDED_BY(idx_mu_)= 0;
  size_t next_read_idx_ ABSL_GUARDED_BY(idx_mu_) = 0;
  bool buffer_not_full_ ABSL_GUARDED_BY(idx_mu_) = true;
  bool stop_ ABSL_GUARDED_BY(idx_mu_) = false;

  // Created and destroyed at table registration/deregistration.
  std::unique_ptr<internal::Thread> worker_ = nullptr;

  absl::FixedArray<Request> buffer_;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_TABLE_EXTENSIONS_ASYNC_BASE_H_
