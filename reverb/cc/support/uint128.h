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

#ifndef REVERB_CC_SUPPORT_UINT128_H_
#define REVERB_CC_SUPPORT_UINT128_H_

#include "absl/numeric/int128.h"
#include "reverb/cc/schema.pb.h"

namespace deepmind {
namespace reverb {

inline Uint128 Uint128ToMessage(const absl::uint128& value) {
  Uint128 message;
  message.set_high(absl::Uint128High64(value));
  message.set_low(absl::Uint128Low64(value));
  return message;
}

inline absl::uint128 MessageToUint128(const Uint128& message) {
  return absl::MakeUint128(message.high(), message.low());
}

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_UINT128_H_
