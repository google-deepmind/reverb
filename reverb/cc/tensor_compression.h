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

#ifndef LEARNING_DEEPMIND_REPLAY_REVERB_TENSOR_COMPRESSION_H_
#define LEARNING_DEEPMIND_REPLAY_REVERB_TENSOR_COMPRESSION_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace deepmind {
namespace reverb {

// Delta encodes INT8,16,32,64 and UINT8,16,32,64 tensors of dimensions >= 2.
// The first dimension is assumed to be the time step and each timestep will be
// encoded as follows: output[i] = input[i] - input[i-1]. For encoding
// `encode=true` should be passed, for decoding `encode=false`.
tensorflow::Tensor DeltaEncode(const tensorflow::Tensor& tensor, bool encode);

// Applies `DeltaEncode` on a vector of tensors.
std::vector<tensorflow::Tensor> DeltaEncodeList(
    const std::vector<tensorflow::Tensor>& tensors, bool encode);

// Compresses a Tensor with Zippy. The resulting `proto` must be read with
// `DecompressTensorFromProto`. Note that string tensors are not compressed.
absl::Status CompressTensorAsProto(const tensorflow::Tensor& tensor,
                                   tensorflow::TensorProto* proto);

// Assumes that the TensorProto was built by calling `CompressTensorAsProto`.
absl::StatusOr<tensorflow::Tensor> DecompressTensorFromProto(
    const tensorflow::TensorProto& proto);

template <typename T>
struct UnsignedType {
  static_assert(
      tensorflow::kDataTypeIsUnsigned.Contains(
          tensorflow::DataTypeToEnum<T>::value),
      "Attempt to treat signed data type as unsigned.  Perhaps a new integer "
      "type was added to TensorFlow's TF_CALL_INTEGRAL_TYPES?  Please extend "
      "UnsignedType specializations for this new data type.");
  typedef T Type;
};

#define REVERB_CREATE_UNSIGNED_TYPE(S, U) \
  template <>                             \
  struct UnsignedType<S> {                \
    typedef U Type;                       \
  };

REVERB_CREATE_UNSIGNED_TYPE(tensorflow::int8, tensorflow::uint8)
REVERB_CREATE_UNSIGNED_TYPE(tensorflow::int16, tensorflow::uint16)
REVERB_CREATE_UNSIGNED_TYPE(tensorflow::int32, tensorflow::uint32)
REVERB_CREATE_UNSIGNED_TYPE(tensorflow::int64, tensorflow::uint64)

#undef REVERB_CREATE_UNSIGNED_TYPE

}  // namespace reverb
}  // namespace deepmind

#endif  // LEARNING_DEEPMIND_REPLAY_REVERB_TENSOR_COMPRESSION_H_
