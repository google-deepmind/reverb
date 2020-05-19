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
#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/schema.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {

class Table;

constexpr absl::Duration kDefaultTimeout = absl::InfiniteDuration();

// Maximum number of events in the ciruclar buffer owned by the StatsManager.
constexpr size_t kEventHistoryBufferSize = 1000000;

// Details about the waiting time for a call to the RateLimiter.
struct RateLimiterEvent {
  // Unique identifier of the call. The ID is created by incrementing the most
  // recently used ID.
  size_t id;

  // Time when the rate limiter recieved the call.
  absl::Time start;

  // The time the rate limiter spent waiting for "permission" to allow the call.
  // If set to ZeroDuration then the call was NOT BLOCKED AT ALL.
  absl::Duration blocked_for;
};

struct RateLimiterEventHistory {
  std::vector<RateLimiterEvent> insert;
  std::vector<RateLimiterEvent> sample;
};

// RateLimiter manages the data throughput for a `Table` by blocking
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

  // Configuration and call stats of the limiter.
  RateLimiterInfo Info(absl::Mutex* mu) const ABSL_SHARED_LOCKS_REQUIRED(mu);

  // Creates a copy of all COMPLETED events created since (inclusive)
  // `min_X_event_id`.
  RateLimiterEventHistory GetEventHistory(absl::Mutex* mu,
                                          size_t min_insert_event_id,
                                          size_t min_sample_event_id) const
      ABSL_SHARED_LOCKS_REQUIRED(mu);

 private:
  friend class Table;
  // `Table` calls these methods on construction and destruction.
  tensorflow::Status RegisterTable(Table* table);
  void UnregisterTable(absl::Mutex* mu, Table* table) ABSL_LOCKS_EXCLUDED(mu);

  // Checks if sample and insert operations can proceed and if so calls `Signal`
  // on respective `CondVar`
  void MaybeSignalCondVars(absl::Mutex* mu) ABSL_SHARED_LOCKS_REQUIRED(mu);

  // Returns Cancelled-status if `Cancel` have been called.
  tensorflow::Status CheckIfCancelled() const;

  // Pointer to the table. We expect this to be available (if set), since it's
  // set by a Table calling RegisterTable(this) after it stores a shared_ptr to
  // this RateLimiter;.
  Table* table_ = nullptr;

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

  // The StatsManager maintains a circular buffer of `RateLimiterEvent` and a
  // set of all time stats for calls of a single type (sample/insert).
  class StatsManager {
   public:
    StatsManager();

    // ScopedEvent automatically marks the event as completed when it goes out
    // of scope.
    class ScopedEvent {
     public:
      ScopedEvent(StatsManager* parent, RateLimiterEvent* event);

      // Should be called to indicate that the event was blocked for any time at
      // all. If this is never called then `blocked_for` will remain as
      // ZeroDuration, ignoring the actual wall time.
      void set_was_blocked();

      ~ScopedEvent();

     private:
      StatsManager* parent_;
      RateLimiterEvent* event_;
      bool was_blocked_;
    };

    // Creates an event using the current time as `start`. The event object is
    // owned by the StatsManager and the pointer must only be used within the
    // RateLimiter.
    ScopedEvent CreateEvent(absl::Mutex* mu) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Marks the event as completed by removing it from `active_` and
    // updating the summary metrics. This method should only be called by
    // ScopedEvent.
    void CompleteEvent(RateLimiterEvent* event);

    // Encode the current state as a `RateLimiterCallStats`-proto.
    void ToProto(absl::Mutex* mu, RateLimiterCallStats* proto) const
        ABSL_SHARED_LOCKS_REQUIRED(mu);

    // Creates a copy of all events starting from `min_event_id` until (but not
    // including) the oldest active event. If no events are currently active
    // all events from `min_event_id` are included in the copy.
    std::vector<RateLimiterEvent> GetEventHistory(absl::Mutex* mu,
                                                  size_t min_event_id) const
        ABSL_SHARED_LOCKS_REQUIRED(mu);

   private:
    // Preallocated buffer of events to avoid allocation while holding the lock
    // on the parent table. The size of the FIXED SIZE
    // (`kEventHistoryBufferSize`) container is set in the constructor of
    // StatsManager.
    absl::FixedArray<RateLimiterEvent> events_;

    // Event IDs are incremented with each created events. Since no concurrent
    // operations are possible we can safely assume that events with larger IDs
    // were started after events with smaller IDs.
    size_t next_event_id_;

    // IDs of events that currently are "active". An event is "active" until
    // for as long as the call is blocked. If no blocking occurs then the event
    // will be added and removed from the set while holding the lock and thus
    // it is never possible to observe an active event that weren't blocked by
    // the rate limiter.
    absl::flat_hash_set<size_t> active_;

    // Number of calls that have been completed.
    int64_t completed_;

    // Number of calls that were blocked for any time at all.
    int64_t limited_;

    // The total time spent waiting for the all the blocked COMPLETED calls.
    // Note that concurrent calls are counted independently so the value can be
    // much larger than the "wall time" since the rate limiter was created.
    absl::Duration total_wait_;
  };

  // Summary statistics and a (large) buffers of recent events.
  StatsManager insert_stats_;
  StatsManager sample_stats_;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_RATE_LIMITER_H_
