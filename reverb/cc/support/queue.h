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

namespace deepmind {
namespace reverb {
namespace internal {

// Thread-safe and closable queue with fixed capacity (buffered channel).
//
// A call to `Push()` inserts an item while `Pop()` removes and retrieves an
// item in fifo order. Once the maximum capacity has been reached, calls to
// `Push()` block until the queue is no longer full. Similarly, `Pop()` blocks
// if there are no items in the queue. `Close()` can be called to unblock any
// pending and future calls to `Push()` and `Pop()`.
//
// Note: This implementation is only intended for a single producer and single
// consumer use case, where Close() is called by the consumer. The
// implementation has not been tested for other use cases!
template <typename T>
class Queue {
 public:
  // `capacity` is the maximum number of elements which the queue can hold.
  explicit Queue(int capacity)
      : buffer_(capacity), size_(0), index_(0), closed_(false) {}

  // Closes the queue. All pending and future calls to `Push()` and `Pop()` are
  // unblocked and return false without performing the operation. Additional
  // calls of Close after the first one have no effect.
  void Close() {
    absl::MutexLock lock(&mu_);
    closed_ = true;
  }

  // Pushes an item to the queue. Blocks if the queue has reached `capacity`. On
  // success, `true` is returned. If the queue is closed, `false` is returned.
  bool Push(T x) {
    absl::MutexLock lock(&mu_);
    mu_.Await(absl::Condition(
        +[](Queue* q) { return q->closed_ || q->size_ < q->buffer_.size(); },
        this));
    if (closed_) return false;
    buffer_[(index_ + size_) % buffer_.size()] = std::move(x);
    ++size_;
    return true;
  }

  // Removes an element from the queue and move-assigns it to *item. Blocks if
  // the queue is empty. On success, `true` is returned. If the queue was
  // closed, `false` is returned.
  bool Pop(T* item) {
    absl::MutexLock lock(&mu_);
    mu_.Await(absl::Condition(
        +[](Queue* q) { return q->closed_ || q->size_ > 0; }, this));
    if (closed_) return false;
    *item = std::move(buffer_[index_]);
    index_ = (index_ + 1) % buffer_.size();
    --size_;
    return true;
  }

  // Current number of elements.
  int size() const {
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
};

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_QUEUE_H_
