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

#ifndef REVERB_CC_RATE_LIMITER_H_
#define REVERB_CC_RATE_LIMITER_H_

#include <string>

#include <cstdint>
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/schema.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {

class PriorityTable;

constexpr absl::Duration kDefaultTimeout = absl::InfiniteDuration();

// RateLimiter manages the data throughput for a PriorityTable by blocking
// sample or insert calls if the ratio between the two deviates too much from
// the ratio specified by `samples_per_insert`.
class RateLimiter {
 public:
  RateLimiter(double samples_per_insert, int64_t min_size_to_sample,
              double min_diff, double max_diff);

  // Construct and restore a RateLimiter from a previous checkpoint.
  explicit RateLimiter(const RateLimiterCheckpoint& checkpoint);

  // Waits until the insert operation can proceed without violating the
  // conditions of the rate limiter.
  //
  // The state is not modified as the caller must first check that the operation
  // is still an insert op (while waiting the item may be inserted by another
  // thread and thus the operation now is an update). If the operation remains
  // an insert then `Insert` must be called to commit the state change.
  tensorflow::Status AwaitCanInsert(absl::Mutex* mu,
                                    absl::Duration timeout = kDefaultTimeout)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Waits until the sample operation can proceed without violating the
  // conditions of the rate limiter. If the condition is fulfilled before the
  // timeout expires or `Cancel` called then the state is updated.
  tensorflow::Status AwaitAndFinalizeSample(
      absl::Mutex* mu, absl::Duration timeout = kDefaultTimeout)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Register that an item has been inserted into the table. Caller must call
  // `AwaitCanInsert` before calling this method without releasing the lock in
  // between.
  void Insert(absl::Mutex* mu) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Register that an item have been deleted from the table.
  void Delete(absl::Mutex* mu) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Register that the table has been fully reset.
  void Reset(absl::Mutex* mu) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Unblocks any `Await` calls with a Cancelled-status.
  void Cancel(absl::Mutex* mu) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Returns true iff the current state would allow for `num_samples` to be
  // sampled. Dies if `num_samples` is < 1.
  bool CanSample(absl::Mutex* mu, int num_samples) const
      ABSL_SHARED_LOCKS_REQUIRED(mu);

  // Returns true iff the current state would allow for `num_inserts` to be
  // inserted. Dies if `num_inserts` is < 1.
  bool CanInsert(absl::Mutex* mu, int num_inserts) const
      ABSL_SHARED_LOCKS_REQUIRED(mu);

  // Creates a checkpoint of the current state for the rate limiter.
  RateLimiterCheckpoint CheckpointReader(absl::Mutex* mu) const
      ABSL_SHARED_LOCKS_REQUIRED(mu);

  // Configuration details of the limiter.
  RateLimiterInfo info() const;

 private:
  friend class PriorityTable;
  // PriorityTable calls these methods on construction and destruction.
  tensorflow::Status RegisterPriorityTable(PriorityTable* table);
  void UnregisterPriorityTable(absl::Mutex* mu, PriorityTable* table)
      ABSL_LOCKS_EXCLUDED(mu);

  // Checks if sample and insert operations can proceed and if so calls `Signal`
  // on respective `CondVar`
  void MaybeSignalCondVars(absl::Mutex* mu) ABSL_SHARED_LOCKS_REQUIRED(mu);

  // Returns Cancelled-status if `Cancel` have been called.
  tensorflow::Status CheckIfCancelled() const;

  // Pointer to the priority table.  We expect this to be available (if set),
  // since it's set by a PriorityTable calling RegisterPriorityTable(this) after
  // it stores a shared_ptr to this RateLimiter;.
  PriorityTable* priority_table_ = nullptr;

  // The desired ratio between sample ops and insert operations. This can be
  // interpreted as the average number of times each item is sampled during
  // its total lifetime.
  const double samples_per_insert_;

  // The minimum and maximum values the cursor is allowed to reach. The cursor
  // value is calculated as `insert_count_ * samples_per_insert_ -
  // sample_count_`. If the value would go beyond these limits then the call is
  // blocked until it can proceed without violating the constraints.
  const double min_diff_;
  const double max_diff_;

  // The minimum number of items that must exist in the distribution for samples
  // to be allowed.
  const int64_t min_size_to_sample_;

  // Total number of items inserted into table.
  int64_t inserts_;

  // Total number of times any item has been sampled from the table.
  int64_t samples_;

  // Total number of items that has been deleted from the table.
  int64_t deletes_;

  // Whether `Cancel` has been called.
  bool cancelled_;

  // Signal called on respective cv if operation can proceed after state change.
  absl::CondVar can_insert_cv_;
  absl::CondVar can_sample_cv_;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_RATE_LIMITER_H_
