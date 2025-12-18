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

#ifndef REVERB_CC_SUPPORT_CLEANUP_H_
#define REVERB_CC_SUPPORT_CLEANUP_H_

#include <memory>

#include "absl/base/attributes.h"

namespace deepmind {
namespace reverb {
namespace internal {

// Calls provided callback when object destroyed.
template <typename Callback>
class Cleanup {
 public:
  explicit Cleanup(Callback&& callback)
      : callback_(std::forward<Callback>(callback)) {}
  ~Cleanup() { callback_(); }

  // Cleanup is neither copyable nor movable.
  Cleanup(const Cleanup&) = delete;
  Cleanup& operator=(const Cleanup&) = delete;

 private:
  Callback callback_;
};

template <typename Callback>
ABSL_MUST_USE_RESULT std::unique_ptr<Cleanup<Callback>> MakeCleanup(
    Callback&& callback) {
  return std::make_unique<Cleanup<Callback>>(std::forward<Callback>(callback));
}

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_CLEANUP_H_
