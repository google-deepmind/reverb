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

#include "reverb/cc/tensor_compression.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/snappy.h"
#include "tensorflow/compiler/xla/tsl/platform/status.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"

namespace deepmind {
namespace reverb {
namespace {

template <typename T>
tensorflow::Tensor DeltaEncode(const tensorflow::Tensor& tensor, bool encode) {
  tensorflow::Tensor output(tensor.dtype(), tensor.shape());

  tensorflow::Tensor tensor_reinterpret;
  TF_CHECK_OK(tensor_reinterpret.BitcastFrom(
      tensor, tensorflow::DataTypeToEnum<T>::v(), tensor.shape()));

  tensorflow::Tensor output_reinterpret;
  TF_CHECK_OK(output_reinterpret.BitcastFrom(
      output, tensorflow::DataTypeToEnum<T>::v(), output.shape()));

  auto src = tensor_reinterpret.flat_outer_dims<T>();
  auto dst = output_reinterpret.flat_outer_dims<T>();
  for (int j = 0; j < src.dimension(1); j++) {
    dst(0, j) = src(0, j);
  }
  for (int i = 1; i < src.dimension(0); i++) {
    for (int j = 0; j < src.dimension(1); j++) {
      dst(i, j) = src(i, j) + (encode ? -src(i - 1, j) : dst(i - 1, j));
    }
  }
  return output;
}

bool IsSupported(const tensorflow::DataType data_type) {
  switch (data_type) {
    case tensorflow::DT_FLOAT:
    case tensorflow::DT_DOUBLE:
    case tensorflow::DT_INT32:
    case tensorflow::DT_UINT8:
    case tensorflow::DT_INT16:
    case tensorflow::DT_INT8:
    case tensorflow::DT_STRING:
    case tensorflow::DT_COMPLEX64:
    case tensorflow::DT_INT64:
    case tensorflow::DT_BOOL:
    case tensorflow::DT_QINT8:
    case tensorflow::DT_QUINT8:
    case tensorflow::DT_QINT32:
    case tensorflow::DT_BFLOAT16:
    case tensorflow::DT_QINT16:
    case tensorflow::DT_QUINT16:
    case tensorflow::DT_UINT16:
    case tensorflow::DT_COMPLEX128:
    case tensorflow::DT_HALF:
    // DT_RESOURCE and DT_VARIANT are not supported.
    case tensorflow::DT_UINT32:
    case tensorflow::DT_UINT64:
    case tensorflow::DT_FLOAT8_E5M2:
    case tensorflow::DT_FLOAT8_E4M3FN:
    case tensorflow::DT_INT4:
    case tensorflow::DT_UINT4:
      return true;
    default:
      return false;
  }
}

}  // namespace

tensorflow::Tensor DeltaEncode(const tensorflow::Tensor& tensor, bool encode) {
  if (tensor.dims() < 2) return tensor;

  switch (tensor.dtype()) {
#define DELTA_ENCODE(T)                      \
  case tensorflow::DataTypeToEnum<T>::value: \
    return DeltaEncode<UnsignedType<T>::Type>(tensor, encode);
    TF_CALL_INTEGRAL_TYPES(DELTA_ENCODE)
#undef DELTA_ENCODE
    default:
      return tensor;
  }
}

std::vector<tensorflow::Tensor> DeltaEncodeList(
    const std::vector<tensorflow::Tensor>& tensors, bool encode) {
  std::vector<tensorflow::Tensor> outputs;
  outputs.reserve(tensors.size());
  for (const tensorflow::Tensor& tensor : tensors) {
    outputs.push_back(DeltaEncode(tensor, encode));
  }
  return outputs;
}

absl::Status CompressTensorAsProto(const tensorflow::Tensor& tensor,
                                   tensorflow::TensorProto* proto) {
  if (!IsSupported(tensor.dtype())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Tensor of dtype ", tensorflow::DataTypeString(tensor.dtype()),
        " is not supported for compression."));
  }

  if (tensor.dtype() == tensorflow::DT_STRING) {
    tensor.AsProtoTensorContent(proto);
    return absl::OkStatus();
  } else {
    proto->set_dtype(tensor.dtype());
    tensor.shape().AsProto(proto->mutable_tensor_shape());
    SnappyCompressFromString(tensor.tensor_data(),
                             proto->mutable_tensor_content());
    return absl::OkStatus();
  }
}

absl::StatusOr<tensorflow::Tensor> DecompressTensorFromProto(
    const tensorflow::TensorProto& proto) {
  if (!IsSupported(proto.dtype())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Tensor of dtype ", tensorflow::DataTypeString(proto.dtype()),
        " is not supported for decompression."));
  }

  if (proto.dtype() == tensorflow::DT_STRING) {
    tensorflow::Tensor tensor;
    REVERB_CHECK(tensor.FromProto(proto));
    return tensor;
  } else {
    tensorflow::Tensor tensor(proto.dtype(),
                              tensorflow::TensorShape(proto.tensor_shape()));
    const auto& tensor_content = proto.tensor_content();
    SnappyUncompressToString(tensor_content, tensor.tensor_data().size(),
                             const_cast<char*>(tensor.tensor_data().data()));
    return tensor;
  }
}

}  // namespace reverb
}  // namespace deepmind
