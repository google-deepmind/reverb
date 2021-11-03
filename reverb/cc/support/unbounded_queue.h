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

#ifndef REVERB_CC_SUPPORT_UNBOUNDED_QUEUE_H_
#define REVERB_CC_SUPPORT_UNBOUNDED_QUEUE_H_

#include <queue>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

namespace deepmind {
namespace reverb {
namespace internal {

// Thread-safe and closable queue.
//
// A call to `Push()` inserts an item while `Pop` removes and retrieves an
// item in fifo order. `Pop` blocks if there are no items in the queue.
// `Close` can be called to unblock any pending and future calls to `Pop`.
// When `SetLastItemPushed` is called, all pending and future calls to `Push`
// will return immediately just as if `Close` had been called. When the queue is
// empty and `SetLastItemPushed` has been called, then the `Close` is
// automatically called. Note that this can occur with the call to
// `SetLastItemPushed` or with subsequent calls to `Pop`
//
template <typename T>
class UnboundedQueue {
 public:
  // Closes the queue. All pending and future calls to `Push()` and `Pop()` are
  // unblocked and return false without performing the operation. Additional
  // calls of Close after the first one have no effect.
  void Close() ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    closed_ = true;
  }

  // Pushes an item to the queue. On success, `true` is returned. If the queue
  // is closed, `false` is returned.
  bool Push(T x) ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    if (closed_ || last_item_pushed_) return false;
    buffer_.push(std::move(x));
    size_++;
    return true;
  }

  // Marks that no more items will be pushed to the queue.
  void SetLastItemPushed() ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    last_item_pushed_ = true;
    if (buffer_.empty()) {
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
    mu_.Await(absl::Condition(
        +[](UnboundedQueue* q) { return q->closed_ || q->size_ > 0; }, this));
    if (closed_) return false;
    *item = std::move(buffer_.front());
    buffer_.pop();
    size_--;
    if (buffer_.empty() && last_item_pushed_) {
      closed_ = true;
    }
    return true;
  }

  // Current number of elements.
  int size() const ABSL_LOCKS_EXCLUDED(mu_) {
    absl::ReaderMutexLock lock(&mu_);
    return size_;
  }

 private:
  mutable absl::Mutex mu_;

  std::queue<T> buffer_ ABSL_GUARDED_BY(mu_);

  // Equivalent to buffer_.size(). We keep it explicitly, otherwise
  // absl::DebugOnlyDeadlockCheck() is not able to detect that there are no
  // deadlocks between Push and Pop.
  int size_ ABSL_GUARDED_BY(mu_) = 0;

  // Whether `Close()` was called.
  bool closed_ ABSL_GUARDED_BY(mu_) = false;

  // Whether `SetLastItemPushed()` has been called. When set then push calls are
  // treated the same as if `Closed()` had been called. If set and the queue is
  // empty after a pop call then `closed_` is set.
  bool last_item_pushed_ ABSL_GUARDED_BY(mu_) = false;
};

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_UNBOUNDED_QUEUE_H_
