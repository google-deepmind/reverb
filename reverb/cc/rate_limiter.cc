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

#include <string>
#include <utility>

#include <cstdint>
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/priority_table.h"
#include "reverb/cc/schema.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace deepmind {
namespace reverb {
namespace {

bool WaitAndLog(absl::CondVar* cv, absl::Mutex* mu, absl::Time deadline,
                absl::string_view call_description,
                absl::Duration log_after = absl::Seconds(10)) {
  const auto log_deadline = absl::Now() + log_after;
  if (log_deadline < deadline) {
    if (!cv->WaitWithDeadline(mu, log_deadline)) {
      return false;
    }
    REVERB_LOG(REVERB_INFO) << call_description << " blocked for " << log_after;
  }

  return cv->WaitWithDeadline(mu, deadline);
}
}  // namespace

RateLimiter::RateLimiter(double samples_per_insert, int64_t min_size_to_sample,
                         double min_diff, double max_diff)
    : samples_per_insert_(samples_per_insert),
      min_diff_(min_diff),
      max_diff_(max_diff),
      min_size_to_sample_(min_size_to_sample),
      inserts_(0),
      samples_(0),
      deletes_(0),
      cancelled_(false) {
  REVERB_CHECK_GT(min_size_to_sample, 0);
}

RateLimiter::RateLimiter(const RateLimiterCheckpoint& checkpoint)
    : RateLimiter(/*samples_per_insert=*/checkpoint.samples_per_insert(),
                  /*min_size_to_sample=*/
                  checkpoint.min_size_to_sample(),
                  /*min_diff=*/checkpoint.min_diff(),
                  /*max_diff=*/checkpoint.max_diff()) {
  inserts_ = checkpoint.insert_count();
  samples_ = checkpoint.sample_count();
  deletes_ = checkpoint.delete_count();
}

tensorflow::Status RateLimiter::RegisterPriorityTable(
    PriorityTable* priority_table) {
  if (priority_table_) {
    return tensorflow::errors::FailedPrecondition(
        "Attempting to registering a priority table ", priority_table,
        " (name: ", priority_table->name(), ") with RateLimiter when is ",
        "already registered with this limiter: ", priority_table_,
        " (name: ", priority_table_->name(), ")");
  }
  priority_table_ = priority_table;
  return tensorflow::Status::OK();
}

void RateLimiter::UnregisterPriorityTable(absl::Mutex* mu,
                                          PriorityTable* table) {
  // Keep priority_table_registered_ at its current value to ensure that
  // no one else tries to access state associated with a table that no longer
  // exists.
  REVERB_CHECK_EQ(table, priority_table_)
      << "The wrong PriorityTable attempted to unregister this rate limiter.";
  absl::MutexLock lock(mu);
  Reset(mu);
  priority_table_ = nullptr;
}

tensorflow::Status RateLimiter::AwaitCanInsert(absl::Mutex* mu,
                                               absl::Duration timeout) {
  const auto start = absl::Now();
  const auto deadline = start + timeout;
  while (!cancelled_ && !CanInsert(mu, 1)) {
    if (WaitAndLog(&can_insert_cv_, mu, deadline, "Insert call")) {
      return tensorflow::errors::DeadlineExceeded(
          "timeout exceeded before right to insert was acquired.");
    }
  }
  TF_RETURN_IF_ERROR(CheckIfCancelled());

  return tensorflow::Status::OK();
}

void RateLimiter::Insert(absl::Mutex* mu) {
  inserts_++;
  MaybeSignalCondVars(mu);
}

void RateLimiter::Delete(absl::Mutex* mu) {
  deletes_++;
  MaybeSignalCondVars(mu);
}

void RateLimiter::Reset(absl::Mutex* mu) {
  inserts_ = 0;
  samples_ = 0;
  deletes_ = 0;
  MaybeSignalCondVars(mu);
}

tensorflow::Status RateLimiter::AwaitAndFinalizeSample(absl::Mutex* mu,
                                                       absl::Duration timeout) {
  const auto start = absl::Now();
  const auto deadline = start + timeout;
  while (!cancelled_ && !CanSample(mu, 1)) {
    if (WaitAndLog(&can_sample_cv_, mu, deadline, "Sample call")) {
      return tensorflow::errors::DeadlineExceeded(
          "timeout exceeded before right to sample was acquired.");
    }
  }
  TF_RETURN_IF_ERROR(CheckIfCancelled());

  samples_++;
  MaybeSignalCondVars(mu);
  return tensorflow::Status::OK();
}

bool RateLimiter::CanSample(absl::Mutex*, int num_samples) const {
  REVERB_CHECK_GT(num_samples, 0);
  if (inserts_ - deletes_ < min_size_to_sample_) {
    return false;
  }
  double diff = inserts_ * samples_per_insert_ - samples_ - num_samples;
  return diff >= min_diff_;
}

bool RateLimiter::CanInsert(absl::Mutex*, int num_inserts) const {
  REVERB_CHECK_GT(num_inserts, 0);
  // Until the min size is reached inserts are free to progress.
  if (inserts_ + num_inserts - deletes_ <= min_size_to_sample_) {
    return true;
  }

  double diff = (num_inserts + inserts_) * samples_per_insert_ - samples_;
  return diff <= max_diff_;
}

void RateLimiter::Cancel(absl::Mutex*) {
  cancelled_ = true;
  can_insert_cv_.SignalAll();
  can_sample_cv_.SignalAll();
}

RateLimiterCheckpoint RateLimiter::CheckpointReader(absl::Mutex*) const {
  RateLimiterCheckpoint checkpoint;
  checkpoint.set_samples_per_insert(samples_per_insert_);
  checkpoint.set_min_diff(min_diff_);
  checkpoint.set_max_diff(max_diff_);
  checkpoint.set_min_size_to_sample(min_size_to_sample_);
  checkpoint.set_sample_count(samples_);
  checkpoint.set_insert_count(inserts_);
  checkpoint.set_delete_count(deletes_);

  return checkpoint;
}

RateLimiterInfo RateLimiter::info() const {
  RateLimiterInfo info_proto;
  info_proto.set_samples_per_insert(samples_per_insert_);
  info_proto.set_min_diff(min_diff_);
  info_proto.set_max_diff(max_diff_);
  info_proto.set_min_size_to_sample(min_size_to_sample_);
  return info_proto;
}

tensorflow::Status RateLimiter::CheckIfCancelled() const {
  if (!cancelled_) return tensorflow::Status::OK();
  return tensorflow::errors::Cancelled("RateLimiter has been cancelled");
}

void RateLimiter::MaybeSignalCondVars(absl::Mutex* mu) {
  if (CanInsert(mu, 1)) can_insert_cv_.Signal();
  if (CanSample(mu, 1)) can_sample_cv_.Signal();
}

}  // namespace reverb
}  // namespace deepmind
