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
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace deepmind {
namespace reverb {
namespace internal {

// Thread-safe and closable queue with fixed capacity (buffered channel).
//
// A call to `Push()` inserts an item while `Pop` removes and retrieves an
// item in fifo order. Once the maximum capacity has been reached, calls to
// `Push` block until the queue is no longer full. Similarly, `Pop` blocks
// if there are no items in the queue. `Close` can be called to unblock any
// pending and future calls to `Push` and `Pop`.
//
// When `SetLastItemPushed` is called, all pending and future calls to `Push`
// will return immediately just as if `Close` had been called. When the queue is
// empty and `SetLastItemPushed` has been called, then the `Close` is
// automatically called. Note that this can occur with the call to
// `SetLastItemPushed` or with subsequent calls to `Pop`
//
template <typename T>
class Queue {
 public:
  // `capacity` is the maximum number of elements which the queue can hold.
  explicit Queue(int capacity)
      : buffer_(capacity),
        size_(0),
        index_(0),
        closed_(false),
        last_item_pushed_(false) {}

  // Closes the queue. All pending and future calls to `Push()` and `Pop()` are
  // unblocked and return false without performing the operation. Additional
  // calls of Close after the first one have no effect.
  void Close() ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    closed_ = true;
  }

  // Pushes an item to the queue. Blocks if the queue has reached `capacity`. On
  // success, `true` is returned. If the queue is closed, `false` is returned.
  bool Push(T x) ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    mu_.Await(absl::Condition(
        +[](Queue* q) {
          return q->closed_ || q->last_item_pushed_ ||
                 q->size_ < q->buffer_.size();
        },
        this));
    if (closed_ || last_item_pushed_) return false;
    buffer_[(index_ + size_) % buffer_.size()] = std::move(x);
    ++size_;
    return true;
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
  tensorflow::Status PopBatch(int batch_size, absl::Duration timeout,
                              std::vector<T>* out) {
    if (batch_size > buffer_.size()) {
      return tensorflow::errors::InvalidArgument("Batch size (", batch_size,
                                                 ") must be <= of queue size (",
                                                 buffer_.size(), ").");
    }

    auto trigger = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return closed_ || size_ >= batch_size || last_item_pushed_;
    };

    absl::MutexLock lock(&mu_);
    if (!mu_.AwaitWithTimeout(absl::Condition(&trigger), timeout) &&
        !last_item_pushed_ && !closed_) {
      return tensorflow::errors::DeadlineExceeded(
          "Timeout exceeeded before ", batch_size, " items observed in queue.");
    }

    if (closed_) {
      return tensorflow::errors::Cancelled("Queue is closed.");
    }

    if (last_item_pushed_) {
      return tensorflow::errors::ResourceExhausted(
          "The last item have been pushed to the queue and the current size (",
          size_, ") is less than the batch size (", batch_size, ").");
    }

    for (int i = 0; i < batch_size; i++) {
      out->push_back(std::move(buffer_[index_]));
      index_ = (index_ + 1) % buffer_.size();
      --size_;
    }

    if (size_ == 0 && last_item_pushed_) {
      closed_ = true;
    }

    return tensorflow::Status::OK();
  }

  tensorflow::Status PopBatch(int batch_size, std::vector<T>* out) {
    return PopBatch(batch_size, absl::InfiniteDuration(), out);
  }

  // Marks that no more items will be pushed to the queue.
  void SetLastItemPushed() ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    last_item_pushed_ = true;
    if (size_ == 0) {
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
        +[](Queue* q) { return q->closed_ || q->size_ > 0; }, this));
    if (closed_) return false;
    *item = std::move(buffer_[index_]);
    index_ = (index_ + 1) % buffer_.size();
    --size_;
    if (size_ == 0 && last_item_pushed_) {
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

  // Circular buffer. Initialized with fixed size `capacity_`.
  std::vector<T> buffer_ ABSL_GUARDED_BY(mu_);

  // Current number of elements.
  int size_ ABSL_GUARDED_BY(mu_);

  // Index of the beginning of the queue in the circular buffer.
  int index_ ABSL_GUARDED_BY(mu_);

  // Whether `Close()` was called.
  bool closed_ ABSL_GUARDED_BY(mu_);

  // Whether `SetLastItemPushed()` has been called. When set then push calls are
  // treated the same as if `Closed()` had been called. If set and the queue is
  // empty after a pop call then `closed_` is set.
  bool last_item_pushed_ ABSL_GUARDED_BY(mu_);
};

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_QUEUE_H_
