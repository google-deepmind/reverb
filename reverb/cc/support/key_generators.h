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

#ifndef REVERB_CC_SUPPORT_KEY_GENERATORS_H_
#define REVERB_CC_SUPPORT_KEY_GENERATORS_H_

#include <cstdint>
#include <limits>

#include "absl/random/random.h"

namespace deepmind::reverb::internal {

class KeyGenerator {
 public:
  virtual ~KeyGenerator() = default;
  virtual uint64_t Generate() = 0;
};

class UniformKeyGenerator : public KeyGenerator {
 public:
  uint64_t Generate() override {
    return absl::Uniform<uint64_t>(bit_gen_, 0,
                                   std::numeric_limits<uint64_t>::max());
  }

 private:
  absl::BitGen bit_gen_;
};

}  // namespace deepmind::reverb::internal

#endif  // REVERB_CC_SUPPORT_KEY_GENERATORS_H_
