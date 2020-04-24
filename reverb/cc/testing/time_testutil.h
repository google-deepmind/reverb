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

#ifndef REVERB_CC_TESTING_TIME_TESTUTIL_H_
#define REVERB_CC_TESTING_TIME_TESTUTIL_H_

#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace deepmind {
namespace reverb {
namespace test {

template <typename F>
void WaitFor(F&& exit_criteria_fn, const absl::Duration& wait_duration,
             int max_iteration) {
  for (int retries = 0; !exit_criteria_fn() && retries < max_iteration;
       ++retries) {
    absl::SleepFor(wait_duration);
  }
}

}  // namespace test
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_TESTING_TIME_TESTUTIL_H_
