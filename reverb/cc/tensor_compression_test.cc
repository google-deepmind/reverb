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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/testing/tensor_testutil.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/platform/tstring.h"

namespace deepmind {
namespace reverb {
namespace {

using ::testing::HasSubstr;
using ::absl_testing::StatusIs;

template <typename T>
void EncodeMatchesDecodeT() {
  tensorflow::Tensor tensor(tensorflow::DataTypeToEnum<T>::v(),
                            tensorflow::TensorShape({16, 37, 6}));
  tensor.flat<T>().setRandom();
  tensorflow::Tensor encoded = DeltaEncode(tensor, true);
  tensorflow::Tensor decoded = DeltaEncode(encoded, false);
  test::ExpectTensorEqual<T>(tensor, decoded);
}

TEST(TensorCompressionTest, EncodeMatchesDecode) {
#define ENCODE_MATCHES_DECODE(T) EncodeMatchesDecodeT<T>();
  TF_CALL_INTEGRAL_TYPES(ENCODE_MATCHES_DECODE)
#undef ENCODE_MATCHES_DECODE
  EncodeMatchesDecodeT<float>();
  EncodeMatchesDecodeT<double>();
  EncodeMatchesDecodeT<bool>();
}

TEST(TensorCompressionTest, EncodeListMatchesDecode) {
  tensorflow::Tensor tensor(tensorflow::DT_INT32,
                            tensorflow::TensorShape({16, 37, 6}));
  tensor.flat<int>().setRandom();
  std::vector<tensorflow::Tensor> tensors{tensor, tensor};
  std::vector<tensorflow::Tensor> encoded = DeltaEncodeList(tensors, true);
  std::vector<tensorflow::Tensor> decoded = DeltaEncodeList(encoded, false);
  EXPECT_EQ(tensors.size(), decoded.size());
  for (int i = 0; i < tensors.size(); i++) {
    test::ExpectTensorEqual<int>(tensors[i], decoded[i]);
  }
}

TEST(TensorCompressionTest, StringTensor) {
  tensorflow::Tensor tensor(tensorflow::DT_STRING,
                            tensorflow::TensorShape({2}));
  tensor.flat<tensorflow::tstring>()(0) = "hello";
  tensor.flat<tensorflow::tstring>()(1) = "world";

  tensorflow::TensorProto proto;
  REVERB_ASSERT_OK(CompressTensorAsProto(tensor, &proto));

  REVERB_ASSERT_OK_AND_ASSIGN(tensorflow::Tensor result,
                       DecompressTensorFromProto(proto));
  test::ExpectTensorEqual<tensorflow::tstring>(tensor, result);
}

TEST(TensorCompressionTest, NonStringTensor) {
  tensorflow::Tensor tensor(tensorflow::DT_INT32,
                            tensorflow::TensorShape({2, 2}));
  tensor.flat<int>().setRandom();

  tensorflow::TensorProto proto;
  REVERB_ASSERT_OK(CompressTensorAsProto(tensor, &proto));

  REVERB_ASSERT_OK_AND_ASSIGN(tensorflow::Tensor result,
                       DecompressTensorFromProto(proto));
  test::ExpectTensorEqual<int>(tensor, result);
}

TEST(TensorCompressionTest, NonStringTensorWithDeltaEncoding) {
  tensorflow::Tensor tensor(tensorflow::DT_INT32,
                            tensorflow::TensorShape({2, 2}));
  tensor.flat<int>().setRandom();

  tensorflow::TensorProto proto;
  REVERB_ASSERT_OK(CompressTensorAsProto(DeltaEncode(tensor, true), &proto));
  REVERB_ASSERT_OK_AND_ASSIGN(tensorflow::Tensor result,
                       DecompressTensorFromProto(proto));
  test::ExpectTensorEqual<int>(tensor, DeltaEncode(result, false));
}

TEST(TensorCompressionTest, CompressingVariantNotSupported) {
  tensorflow::Tensor tensor(tensorflow::DT_VARIANT,
                            tensorflow::TensorShape({}));

  tensorflow::Tensor internal(tensorflow::DT_FLOAT,
                              tensorflow::TensorShape({2, 2}));
  internal.flat<float>().setRandom();
  tensor.flat<tensorflow::Variant>()(0) = internal;

  tensorflow::TensorProto proto;
  EXPECT_THAT(CompressTensorAsProto(DeltaEncode(tensor, true), &proto),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("variant is not supported")));
}

TEST(TensorCompressionTest, DecompressingVariantNotSupported) {
  tensorflow::TensorProto proto;
  proto.set_dtype(tensorflow::DT_VARIANT);

  EXPECT_THAT(DecompressTensorFromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("variant is not supported")));
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
