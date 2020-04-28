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

#include <algorithm>
#include <string>
#include <utility>

#include "google/protobuf/duration.pb.h"
#include <cstdint>
#include "absl/base/thread_annotations.h"
#include "absl/container/fixed_array.h"
#include "absl/strings/string_view.h"
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

inline void EncodeAsDurationProto(const absl::Duration& d,
                                  google::protobuf::Duration* proto) {
  proto->set_seconds(absl::ToInt64Seconds(d));
  proto->set_nanos(
      absl::ToInt64Nanoseconds(d - absl::Seconds(proto->seconds())));
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
      cancelled_(false),
      insert_stats_(),
      sample_stats_() {
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
  const auto deadline = absl::Now() + timeout;
  {
    auto event = insert_stats_.CreateEvent(mu);
    while (!cancelled_ && !CanInsert(mu, 1)) {
      event.set_was_blocked();
      if (can_insert_cv_.WaitWithDeadline(mu, deadline)) {
        return tensorflow::errors::DeadlineExceeded(
            "timeout exceeded before right to insert was acquired.");
      }
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
  const auto deadline = absl::Now() + timeout;

  {
    auto event = sample_stats_.CreateEvent(mu);
    while (!cancelled_ && !CanSample(mu, 1)) {
      event.set_was_blocked();
      if (can_sample_cv_.WaitWithDeadline(mu, deadline)) {
        return tensorflow::errors::DeadlineExceeded(
            "timeout exceeded before right to sample was acquired.");
      }
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

tensorflow::Status RateLimiter::CheckIfCancelled() const {
  if (!cancelled_) return tensorflow::Status::OK();
  return tensorflow::errors::Cancelled("RateLimiter has been cancelled");
}

void RateLimiter::MaybeSignalCondVars(absl::Mutex* mu) {
  if (CanInsert(mu, 1)) can_insert_cv_.Signal();
  if (CanSample(mu, 1)) can_sample_cv_.Signal();
}

RateLimiterInfo RateLimiter::Info(absl::Mutex* mu) const {
  RateLimiterInfo info_proto;
  info_proto.set_samples_per_insert(samples_per_insert_);
  info_proto.set_min_diff(min_diff_);
  info_proto.set_max_diff(max_diff_);
  info_proto.set_min_size_to_sample(min_size_to_sample_);
  insert_stats_.ToProto(mu, info_proto.mutable_insert_stats());
  sample_stats_.ToProto(mu, info_proto.mutable_sample_stats());
  return info_proto;
}

RateLimiterEventHistory RateLimiter::GetEventHistory(
    absl::Mutex* mu, size_t min_insert_event_id,
    size_t min_sample_event_id) const {
  return {insert_stats_.GetEventHistory(mu, min_insert_event_id),
          sample_stats_.GetEventHistory(mu, min_sample_event_id)};
}

RateLimiter::StatsManager::StatsManager()
    : events_(kEventHistoryBufferSize),
      next_event_id_(0),
      active_(),
      completed_(0),
      limited_(0),
      total_wait_(absl::ZeroDuration()) {}

RateLimiter::StatsManager::ScopedEvent RateLimiter::StatsManager::CreateEvent(
    absl::Mutex* mu) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) {
  // IDs are incremented with each event and the fixed size of `events_` means
  // that it is theoretically possible for collisions between active events.
  // However, this is EXTREMELY unlikely in practice as it would require
  // `kEventHistoryBufferSize` (very large) calls to be blocked concurrently
  // which would grind the system to a halt long before the buffer is exceeded.
  const size_t id = next_event_id_++;
  events_[id % events_.size()] = {id, absl::Now(), absl::ZeroDuration()};
  active_.insert(id);
  return ScopedEvent(this, &events_[id % events_.size()]);
}

void RateLimiter::StatsManager::CompleteEvent(RateLimiterEvent* event) {
  active_.erase(event->id);
  completed_++;
  limited_ += event->blocked_for > absl::ZeroDuration() ? 1 : 0;
  total_wait_ += event->blocked_for;
}

void RateLimiter::StatsManager::ToProto(absl::Mutex* mu,
                                        RateLimiterCallStats* proto) const
    ABSL_SHARED_LOCKS_REQUIRED(mu) {
  const auto now = absl::Now();

  proto->set_pending(active_.size());
  proto->set_completed(completed_);
  proto->set_limited(limited_);
  EncodeAsDurationProto(total_wait_, proto->mutable_completed_wait_time());
  absl::Duration pending_wait_time;
  for (size_t id : active_) {
    pending_wait_time += now - events_[id % events_.size()].start;
  }
  EncodeAsDurationProto(pending_wait_time, proto->mutable_pending_wait_time());
}

std::vector<RateLimiterEvent> RateLimiter::StatsManager::GetEventHistory(
    absl::Mutex* mu, size_t min_event_id) const ABSL_SHARED_LOCKS_REQUIRED(mu) {
  REVERB_CHECK_LE(min_event_id, next_event_id_);

  if (const auto diff = next_event_id_ - min_event_id; diff >= events_.size()) {
    REVERB_LOG(REVERB_ERROR)
        << "Requested rate limiter events older that the maximum age. Request "
           "will be rewritten to include the last "
        << events_.size() << " events. This mean that (up to) "
        << diff - events_.size() << " events will be ignored";
    min_event_id = next_event_id_ - events_.size();
  }

  // Non inclusive upper limit.
  const size_t max_event_id =
      active_.empty() ? next_event_id_
                      : *std::min_element(active_.begin(), active_.end());
  if (max_event_id < min_event_id + 1) {
    return {};
  }

  std::vector<RateLimiterEvent> copy(max_event_id - min_event_id - 1);
  for (size_t i = 0; i < copy.size(); i++) {
    copy[i] = events_[(min_event_id + i) % events_.size()];
  }

  return copy;
}

RateLimiter::StatsManager::ScopedEvent::ScopedEvent(
    RateLimiter::StatsManager* parent, RateLimiterEvent* event)
    : parent_(parent), event_(event), was_blocked_(false) {}

void RateLimiter::StatsManager::ScopedEvent::set_was_blocked() {
  was_blocked_ = true;
}

RateLimiter::StatsManager::ScopedEvent::~ScopedEvent() {
  if (was_blocked_) {
    event_->blocked_for = absl::Now() - event_->start;
  }
  parent_->CompleteEvent(event_);
}

}  // namespace reverb
}  // namespace deepmind
