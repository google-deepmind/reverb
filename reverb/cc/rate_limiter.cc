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
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/errors.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/table.h"

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
      deletes_(0) {
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

absl::Status RateLimiter::RegisterTable(Table* table) {
  if (table_) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Attempting to registering a table ", absl::Hex(table),
        " (name: ", table->name(), ") with RateLimiter when is ",
        "already registered with this limiter: ", absl::Hex(table_),
        " (name: ", table_->name(), ")"));
  }
  table_ = table;
  return absl::OkStatus();
}

void RateLimiter::UnregisterTable(absl::Mutex* mu, Table* table) {
  REVERB_CHECK_EQ(table, table_)
      << "The wrong Table attempted to unregister this rate limiter.";
  absl::MutexLock lock(mu);
  Reset(mu);
  table_ = nullptr;
}

void RateLimiter::Insert(absl::Mutex* mu) {
  inserts_++;
}

void RateLimiter::Delete(absl::Mutex* mu) {
  deletes_++;
}

void RateLimiter::Reset(absl::Mutex* mu) {
  inserts_ = 0;
  samples_ = 0;
  deletes_ = 0;
}

bool RateLimiter::CanSample(absl::Mutex*, int num_samples) const {
  REVERB_CHECK_GT(num_samples, 0);
  if (inserts_ - deletes_ < min_size_to_sample_) {
    return false;
  }
  double diff = inserts_ * samples_per_insert_ - samples_ - num_samples;
  return diff >= min_diff_;
}

bool RateLimiter::MaybeCommitSample(absl::Mutex* mu) {
  if (!CanSample(mu, 1)) {
    return false;
  }
  samples_++;
  return true;
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

RateLimiterInfo RateLimiter::Info(absl::Mutex* mu) const {
  RateLimiterInfo info_proto = InfoWithoutCallStats();
  info_proto.mutable_insert_stats()->set_completed(inserts_);
  info_proto.mutable_sample_stats()->set_completed(samples_);
  return info_proto;
}

RateLimiterInfo RateLimiter::InfoWithoutCallStats() const {
  RateLimiterInfo info_proto;
  info_proto.set_samples_per_insert(samples_per_insert_);
  info_proto.set_min_diff(min_diff_);
  info_proto.set_max_diff(max_diff_);
  info_proto.set_min_size_to_sample(min_size_to_sample_);
  return info_proto;
}

std::string RateLimiter::DebugString() const {
  return absl::StrCat("RateLimiter(samples_per_insert=", samples_per_insert_,
                      ", min_diff_=", min_diff_, ", max_diff=", max_diff_,
                      ", min_size_to_sample=", min_size_to_sample_, ")");
}

}  // namespace reverb
}  // namespace deepmind
