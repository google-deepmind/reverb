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

#ifndef REVERB_CC_SUPPORT_TF_UTIL_H_
#define REVERB_CC_SUPPORT_TF_UTIL_H_

#include "absl/status/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {

// Converts a tensorflow::Status object to an absl::Status object.
inline absl::Status FromTensorflowStatus(const tensorflow::Status& status) {
  if (status.ok()) {
    return absl::Status();
  } else {
    return absl::Status(static_cast<absl::StatusCode>(status.code()),
                        status.error_message());
  }
}

// Converts an absl::Status object to a tensorflow::Status object.
inline tensorflow::Status ToTensorflowStatus(const absl::Status& status) {
  if (status.ok()) {
    return tensorflow::Status();
  } else {
    return tensorflow::Status(
        static_cast<tensorflow::error::Code>(status.code()), status.message());
  }
}

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_STATUS_UTIL_H_
