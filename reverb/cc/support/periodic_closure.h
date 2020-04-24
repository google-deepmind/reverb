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

#ifndef REVERB_CC_SUPPORT_PERIODIC_CLOSURE_H_
#define REVERB_CC_SUPPORT_PERIODIC_CLOSURE_H_

#include <functional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "reverb/cc/platform/thread.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {
namespace internal {

// PeriodicClosure will periodically call the given closure with a specified
// period in a background thread.  After Start() returns, the thread is
// guaranteed to have started and after Stop() returns, the thread is
// guaranteed to be stopped. Start()/Stop() may be called more than once; each
// pair of calls will result in a new thread being created and subsequently
// destroyed.
//
// PeriodicClosure runs the closure as soon as any previous run both is
// complete and was started more than "interval" earlier.  Thus, runs are
// both serialized, and normally have a period of "interval" if no run
// exceeds the time.
//
// Note that, if the closure takes longer than the interval, then the closure is
// called immediately and the next call is scheduled at `interval` into the
// future. If the interval is 50ms and the first call to the closure takes 75ms
// and all other calls takes 25ms, then the closure will run at: 0ms, 75ms,
// 125ms, 175ms, and so on.
//
// This object is thread-safe.
//
// Example:
//
//   class Foo {
//    public:
//     Foo() : periodic_closure_([this]() { Bar(); },
//                               absl::Seconds(1)) {
//       periodic_closure_.Start();
//     }
//
//     ~Foo() {
//       periodic_closure_.Stop();
//     }
//
//    private:
//     void Bar() { ... }
//
//     PeriodicClosure periodic_closure_;
//   };
//
class PeriodicClosure {
 public:
  PeriodicClosure(std::function<void()> fn, absl::Duration period,
                  std::string name_prefix = "");

  // Dies if `Start` but not `Stop` called.
  ~PeriodicClosure();

  // Starts the background thread that will be calling the closure.
  //
  // Returns InvalidArgument if called more than once.
  tensorflow::Status Start();

  // Waits for active closure call (if any) to complete and joins background
  // thread. Must be called before object is destroyed and `Start` has been
  // called.
  //
  // Returns InvalidArgument if called more than once.
  tensorflow::Status Stop();

  // PeriodicClosure is neither copyable nor movable.
  PeriodicClosure(const PeriodicClosure&) = delete;
  PeriodicClosure& operator=(const PeriodicClosure&) = delete;

 private:
  // Closure called by the background thread.
  const std::function<void()> fn_;

  // The minimum duration between calls to `fn_`.
  const absl::Duration period_;

  // Name prefix assigned to background thread.
  const std::string name_prefix_;

  // Flag to ensure that `Start` and `Stop` is not called multiple times.
  bool stopped_ ABSL_GUARDED_BY(mu_) = false;
  absl::Mutex mu_;

  // Background thread constructed in `Start` and joined in `Stop`.
  std::unique_ptr<Thread> worker_ = nullptr;
};

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_PERIODIC_CLOSURE_H_
