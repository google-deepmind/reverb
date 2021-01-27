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

#include "reverb/cc/errors.h"

#include "absl/strings/match.h"

namespace deepmind {
namespace reverb {
namespace errors {

namespace {

constexpr auto kTimeoutExceededErrorMessage =
    "Rate Limiter: Timeout exceeded before the right to insert was acquired.";

}  // namespace

absl::Status RateLimiterTimeout() {
  return absl::DeadlineExceededError(kTimeoutExceededErrorMessage);
}

bool IsRateLimiterTimeout(absl::Status status) {
  return absl::IsDeadlineExceeded(status) &&
         absl::StrContains(status.message(), kTimeoutExceededErrorMessage);
}

}  // namespace errors
}  // namespace reverb
}  // namespace deepmind
