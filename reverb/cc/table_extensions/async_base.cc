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

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/table.h"
#include "tensorflow/core/platform/errors.h"

namespace deepmind {
namespace reverb {

TableExtensionAsyncBase::TableExtensionAsyncBase(size_t buffer_size)
    : buffer_(buffer_size) {}

tensorflow::Status TableExtensionAsyncBase::RegisterTable(absl::Mutex* mu,
                                                          Table* table) {
  {
    absl::WriterMutexLock lock(&table_mu_);
    if (table_) {
      return tensorflow::errors::FailedPrecondition(
          "Attempting to registering a table ", table,
          " (name: ", table->name(), ") with extension that has already been ",
          "registered with: ", table_, " (name: ", table->name(), ")");
    }
    table_ = table;
  }
  {
    absl::WriterMutexLock lock(&idx_mu_);
    stop_ = false;
  }

  worker_ =
      internal::StartThread("TableExtensionWorker", [this] { RunWorker(); });

  return tensorflow::Status::OK();
}

TableExtensionAsyncBase::~TableExtensionAsyncBase() {
  absl::WriterMutexLock lock(&table_mu_);
  REVERB_CHECK(table_ == nullptr)
      << "Extension destroyed before unregistered from parent table.";
}

void TableExtensionAsyncBase::UnregisterTable(absl::Mutex* mu, Table* table) {
  {
    absl::WriterMutexLock lock(&idx_mu_);
    stop_ = true;
  }
  worker_ = nullptr;  // Joins thread.

  absl::WriterMutexLock lock(&table_mu_);
  REVERB_CHECK_EQ(table, table_)
      << "The wrong Table attempted to unregister this extension.";
  table_ = nullptr;
}

void TableExtensionAsyncBase::OnDelete(absl::Mutex* mu, const TableItem& item) {
  Push({Request::CallType::kDelete, item});
}

void TableExtensionAsyncBase::OnInsert(absl::Mutex* mu, const TableItem& item) {
  Push({Request::CallType::kInsert, item});
}

void TableExtensionAsyncBase::OnReset(absl::Mutex* mu) {
  Push(Request{Request::CallType::kReset});
}

void TableExtensionAsyncBase::OnUpdate(absl::Mutex* mu, const TableItem& item) {
  Push({Request::CallType::kUpdate, item});
}

void TableExtensionAsyncBase::OnSample(absl::Mutex* mu, const TableItem& item) {
  Push({Request::CallType::kSample, item});
}

void TableExtensionAsyncBase::RunWorker() {
  bool stop = false;
  while (!stop) {
    size_t start;
    size_t end;
    {
      absl::MutexLock lock(&idx_mu_);
      idx_mu_.Await(absl::Condition(
          +[](TableExtensionAsyncBase* ext)
               ABSL_EXCLUSIVE_LOCKS_REQUIRED(idx_mu_) {
                 return ext->next_idx_ > ext->next_read_idx_ || ext->stop_;
               },
          this));

      start = next_read_idx_;
      end = next_idx_;
      stop = stop_;
    }

    for (size_t i = start; i < end; ++i) {
      auto& request = buffer_[i % buffer_.size()];
      switch (request.call_type) {
        case Request::CallType::kDelete:
          ApplyOnDelete(request.item);
          break;
        case Request::CallType::kInsert:
          ApplyOnInsert(request.item);
          break;
        case Request::CallType::kReset:
          ApplyOnReset();
          break;
        case Request::CallType::kUpdate:
          ApplyOnUpdate(request.item);
          break;
        case Request::CallType::kSample:
          ApplyOnSample(request.item);
          break;
      }
      request.item.chunks.clear();
    }

    {
      absl::MutexLock lock(&idx_mu_);
      next_read_idx_ = end;
      buffer_not_full_ = true;
    }
  }

  {
    absl::MutexLock lock(&idx_mu_);
    REVERB_CHECK_EQ(next_read_idx_, next_idx_)
        << "Extension shut down before all events handled.";
  }
}

void TableExtensionAsyncBase::WaitUntilReady() {
  absl::MutexLock lock(&idx_mu_);
  size_t next_idx_copy = next_idx_;
  auto cond = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(idx_mu_) {
    return next_read_idx_ >= next_idx_copy;
  };
  idx_mu_.Await(absl::Condition(&cond));
}

void TableExtensionAsyncBase::Push(TableExtensionAsyncBase::Request request) {
  absl::MutexLock lock(&idx_mu_);
  idx_mu_.Await(absl::Condition(&buffer_not_full_));
  buffer_[next_idx_ % buffer_.size()] = std::move(request);
  buffer_not_full_ = ++next_idx_ - next_read_idx_ < buffer_.size();
}

void TableExtensionAsyncBase::ApplyOnDelete(const TableItem& item) {}

void TableExtensionAsyncBase::ApplyOnInsert(const TableItem& item) {}

void TableExtensionAsyncBase::ApplyOnReset() {}

void TableExtensionAsyncBase::ApplyOnUpdate(const TableItem& item) {}

void TableExtensionAsyncBase::ApplyOnSample(const TableItem& item) {}

}  // namespace reverb
}  // namespace deepmind
