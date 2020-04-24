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

#include <string>

#include "gtest/gtest.h"
#include "reverb/cc/testing/tensor_testutil.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"

namespace deepmind {
namespace reverb {
namespace {

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
  CompressTensorAsProto(tensor, &proto);

  tensorflow::Tensor result = DecompressTensorFromProto(proto);
  test::ExpectTensorEqual<tensorflow::tstring>(tensor, result);
}

TEST(TensorCompressionTest, NonStringTensor) {
  tensorflow::Tensor tensor(tensorflow::DT_INT32,
                            tensorflow::TensorShape({2, 2}));
  tensor.flat<int>().setRandom();

  tensorflow::TensorProto proto;
  CompressTensorAsProto(tensor, &proto);

  tensorflow::Tensor result = DecompressTensorFromProto(proto);
  test::ExpectTensorEqual<int>(tensor, result);
}

TEST(TensorCompressionTest, NonStringTensorWithDeltaEncoding) {
  tensorflow::Tensor tensor(tensorflow::DT_INT32,
                            tensorflow::TensorShape({2, 2}));
  tensor.flat<int>().setRandom();

  tensorflow::TensorProto proto;
  CompressTensorAsProto(DeltaEncode(tensor, true), &proto);

  tensorflow::Tensor result = DecompressTensorFromProto(proto);
  test::ExpectTensorEqual<int>(tensor, DeltaEncode(result, false));
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
