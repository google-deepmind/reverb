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

#include "reverb/cc/support/periodic_closure.h"

#include <functional>
#include <string>

#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "reverb/cc/platform/logging.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace deepmind {
namespace reverb {
namespace internal {

PeriodicClosure::~PeriodicClosure() {
  REVERB_CHECK(worker_ == nullptr) << "must be Stop()'d before destructed";
}

PeriodicClosure::PeriodicClosure(std::function<void()> fn,
                                 absl::Duration period, std::string name_prefix)
    : fn_(std::move(fn)),
      period_(period),
      name_prefix_(std::move(name_prefix)) {
  REVERB_CHECK_GE(period_, absl::ZeroDuration()) << "period should be >= 0";
}

tensorflow::Status PeriodicClosure::Start() {
  absl::WriterMutexLock lock(&mu_);
  if (stopped_) {
    return tensorflow::errors::InvalidArgument(
        "PeriodicClosure: Start called after Close");
  }
  if (worker_ != nullptr) {
    return tensorflow::errors::InvalidArgument(
        "PeriodicClosure: Start called when closure already running");
  }
  worker_ = StartThread(name_prefix_, [this] {
    for (auto next_run = absl::Now() + period_; true;) {
      if (mu_.LockWhenWithDeadline(absl::Condition(&stopped_), next_run)) {
        mu_.Unlock();
        return;
      }
      mu_.Unlock();
      next_run = absl::Now() + period_;

      fn_();
    }
  });
  return tensorflow::Status::OK();
}

tensorflow::Status PeriodicClosure::Stop() {
  {
    absl::MutexLock l(&mu_);
    if (stopped_) {
      return tensorflow::errors::InvalidArgument(
          "PeriodicClsoure: Stop called multiple times");
    }
    stopped_ = true;
  }
  worker_ = nullptr;  // Join thread.
  return tensorflow::Status::OK();
}

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind
