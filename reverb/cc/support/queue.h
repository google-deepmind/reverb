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

#ifndef REVERB_CC_SUPPORT_QUEUE_H_
#define REVERB_CC_SUPPORT_QUEUE_H_

#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "reverb/cc/platform/logging.h"

namespace deepmind {
namespace reverb {
namespace internal {

// Thread-safe and closable queue with fixed capacity (buffered channel) and
// pre-reservations.
//
// A call to `Push()` inserts an item while `Pop` removes and retrieves an
// item in fifo order. `Push()` call has to be preceded with `Reserve()` call.
// Once the maximum capacity has been reached, calls to
// `Reserve` block until there is sufficient space in the queue. Similarly,
// `Pop` blocks if there are no items in the queue. `Close` can be called to
// unblock any pending and future calls to `Reserve` and `Pop`.
//
// When `SetLastItemPushed` is called, all pending and future calls to `Reserve`
// will return immediately just as if `Close` had been called. When the queue is
// empty and `SetLastItemPushed` has been called, then the `Close` is
// automatically called. Note that this can occur with the call to
// `SetLastItemPushed` or with subsequent calls to `Pop`.
//
template <typename T>
class Queue {
 public:
  // `capacity` is the maximum number of elements which the queue can hold.
  explicit Queue(int capacity)
      : buffer_(std::max(0, capacity)),
        pushes_(0),
        pops_(0),
        reserved_(0),
        closed_(false),
        last_item_pushed_(false) {}

  // Closes the queue. All pending and future calls to `Reserve()` and `Pop()`
  // are unblocked and return false without performing the operation. Additional
  // calls of Close after the first one have no effect.
  void Close() ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    closed_ = true;
  }

  // Reserves a given number of slots in the queue. Blocks if there is not
  // sufficient space in the queue. On success, `true` is returned.
  // If the queue is closed, `false` is returned.
  bool Reserve(int count) ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);

    auto trigger = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return closed_ || last_item_pushed_ ||
             pushes_ + reserved_ - pops_ + count <= buffer_.size();
    };
    mu_.Await(absl::Condition(&trigger));
    reserved_ += count;
    return !(closed_ || last_item_pushed_);
  }

  // Pushes a batch of items using std::move and then calls `clear` on the input
  // vector.
  // NOTE! Space for all elements of the provided vector must be reserved before
  // calling this method. Failing to do so will trigger death.
  void PushBatch(std::vector<T>* x) {
    absl::MutexLock lock(&mu_);
    REVERB_CHECK_GE(reserved_, x->size())
        << "Space has not been reserved in the queue. Please file a bug to the "
           "Reverb team.";
    reserved_ -= x->size();
    for (auto& i : *x) {
      buffer_[pushes_ % buffer_.size()] = std::move(i);
      ++pushes_;
    }
    x->clear();
  }

  // Exactly the same as the method above, but accepts vector of elements
  // instead of a pointer.
  void PushBatch(std::vector<T> x) {
    PushBatch(&x);
  }

  // Blocks until queue contains at least `batch_size` items then pops and
  // pushes `batch_size` from the queue to `out`.
  //
  // Returns:
  //   OK: If `batch_size` items could be popped before `timeout`.
  //   InvalidArgumentError: if `batch_size` > queue size.
  //   DeadlineExceededError: if timeout exceeded.
  //   ResourceExhaustedError: if SetLastItemPushed called before `batch_size`
  //     items in the queue.
  //   CancelledError: if queue has been closed or SetLastItemPushed called on
  //     an already empty queue.
  //
  absl::Status PopBatch(int batch_size, absl::Duration timeout,
                        std::vector<T>* out) {
    if (batch_size > buffer_.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Batch size (", batch_size,
                       ") must be <= of queue size (", buffer_.size(), ")."));
    }

    auto trigger = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return closed_ || pushes_ - pops_ >= batch_size || last_item_pushed_;
    };

    absl::MutexLock lock(&mu_);
    ScopedIncrement ticket(&num_waiting_to_pop_);

    if (!mu_.AwaitWithTimeout(absl::Condition(&trigger), timeout) &&
        !last_item_pushed_ && !closed_) {
      return absl::DeadlineExceededError(
          absl::StrCat("Timeout exceeded before ", batch_size,
                       " items observed in queue."));
    }

    if (closed_) {
      return absl::CancelledError("Queue is closed.");
    }

    if (last_item_pushed_) {
      return absl::ResourceExhaustedError(absl::StrCat(
          "The last item have been pushed to the queue and the current size (",
          pushes_ - pops_, ") is less than the batch size (", batch_size, ").")
      );
    }

    for (int i = 0; i < batch_size; i++) {
      out->push_back(std::move(buffer_[pops_ % buffer_.size()]));
      pops_++;
    }

    if (pops_ == pushes_ && last_item_pushed_) {
      closed_ = true;
    }

    return absl::OkStatus();
  }

  absl::Status PopBatch(int batch_size, std::vector<T>* out) {
    return PopBatch(batch_size, absl::InfiniteDuration(), out);
  }

  // Marks that no more items will be pushed to the queue.
  void SetLastItemPushed() ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    last_item_pushed_ = true;
    if (pushes_ == pops_) {
      closed_ = true;
    }
  }

  // Removes an element from the queue and move-assigns it to *item. Blocks if
  // the queue is empty. On success, `true` is returned. If the queue was
  // closed, `false` is returned.
  //
  // If called after `SetLastItemPushed` and the final item of the queue is
  // returned then queue is closed.
  bool Pop(T* item) ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    ScopedIncrement ticket(&num_waiting_to_pop_);

    mu_.Await(absl::Condition(+[](Queue* q) {
        return q->closed_ || q->pushes_ > q->pops_; }, this));
    if (closed_) return false;
    *item = std::move(buffer_[pops_ % buffer_.size()]);
    pops_++;
    if (pushes_ == pops_ && last_item_pushed_) {
      closed_ = true;
    }
    return true;
  }

  // Current number of elements.
  int size() const ABSL_LOCKS_EXCLUDED(mu_) {
    absl::ReaderMutexLock lock(&mu_);
    return pushes_ - pops_;
  }

  int num_waiting_to_pop() const ABSL_LOCKS_EXCLUDED(mu_) {
    absl::ReaderMutexLock lock(&mu_);
    return num_waiting_to_pop_;
  }

  int num_pushes() const ABSL_LOCKS_EXCLUDED(mu_) {
    absl::ReaderMutexLock lock(&mu_);
    return pushes_;
  }

 private:
  mutable absl::Mutex mu_;

  // Increments a counter while in scope.
  class ScopedIncrement {
   public:
    ScopedIncrement(int* value) : value_(value) { ++(*value_); }
    ~ScopedIncrement() { --(*value_); }

   private:
    int* value_;
  };

  // Circular buffer. Initialized with fixed size `capacity_`.
  std::vector<T> buffer_ ABSL_GUARDED_BY(mu_);

  // Total number of pushed elements.
  int64_t pushes_ ABSL_GUARDED_BY(mu_);

  // Number of slots reserved for the future pushes.
  int64_t reserved_ ABSL_GUARDED_BY(mu_);

  // Total number of poped elements.
  int64_t pops_ ABSL_GUARDED_BY(mu_);

  // Whether `Close()` was called.
  bool closed_ ABSL_GUARDED_BY(mu_);

  // Whether `SetLastItemPushed()` has been called. When set then push calls are
  // treated the same as if `Closed()` had been called. If set and the queue is
  // empty after a pop call then `closed_` is set.
  bool last_item_pushed_ ABSL_GUARDED_BY(mu_);

  // The number of threads which are currently waiting on the queue.
  int num_waiting_to_pop_ ABSL_GUARDED_BY(mu_) = 0;
} ABSL_CACHELINE_ALIGNED;
static_assert(sizeof(Queue<bool>) >= ABSL_CACHELINE_SIZE,
              "Queue has to take the entire cache line so that its lock is not "
              "colocated with other locks.");

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_QUEUE_H_
