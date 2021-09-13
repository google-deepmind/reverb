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

#include "reverb/cc/support/signature.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace deepmind {
namespace reverb {
namespace internal {
namespace {

using ::deepmind::reverb::testing::CreateProto;
using ::deepmind::reverb::testing::EqualsProto;

tensorflow::StructuredValue MakeLeaf(
    const std::string& name,
    tensorflow::DataType dtype = tensorflow::DT_FLOAT,
    const tensorflow::PartialTensorShape& shape =
        tensorflow::PartialTensorShape()) {
  tensorflow::StructuredValue value;
  tensorflow::TensorSpecProto* tensor_spec = value.mutable_tensor_spec_value();
  tensor_spec->set_name(name);
  tensor_spec->set_dtype(dtype);
  shape.AsProto(tensor_spec->mutable_shape());
  return value;
}

TEST(FlatSignatureFromStructuredValueTest, TensorSpec) {
  tensorflow::StructuredValue value =
      MakeLeaf("leaf", tensorflow::DT_FLOAT, tensorflow::PartialTensorShape());

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromStructuredValue(value, &dtypes_and_shapes);
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(dtypes_and_shapes.has_value());
  EXPECT_EQ(dtypes_and_shapes.value().size(), 1);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "leaf");
  EXPECT_EQ(dtypes_and_shapes.value()[0].dtype, tensorflow::DT_FLOAT);
  EXPECT_EQ(dtypes_and_shapes.value()[0].shape.dims(), -1);
}

TEST(FlatSignatureFromStructuredValueTest, BoundedTensorSpec) {
  tensorflow::StructuredValue value;
  tensorflow::BoundedTensorSpecProto* bounded_tensor_spec =
      value.mutable_bounded_tensor_spec_value();
  bounded_tensor_spec->set_name("leaf");
  bounded_tensor_spec->set_dtype(tensorflow::DT_INT32);
  tensorflow::PartialTensorShape({8}).AsProto(
      bounded_tensor_spec->mutable_shape());
  tensorflow::Tensor(0).AsProtoTensorContent(
      bounded_tensor_spec->mutable_minimum());
  tensorflow::Tensor(255).AsProtoTensorContent(
      bounded_tensor_spec->mutable_maximum());

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromStructuredValue(value, &dtypes_and_shapes);
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(dtypes_and_shapes.has_value());
  EXPECT_EQ(dtypes_and_shapes.value().size(), 1);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "leaf");
  EXPECT_EQ(dtypes_and_shapes.value()[0].dtype, tensorflow::DT_INT32);
  EXPECT_EQ(dtypes_and_shapes.value()[0].shape.dims(), 1);
}

TEST(FlatSignatureFromStructuredValueTest, ListNaming) {
  tensorflow::StructuredValue value;
  *value.mutable_list_value()->add_values() = MakeLeaf("one");
  *value.mutable_list_value()->add_values() = MakeLeaf("two");

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromStructuredValue(value, &dtypes_and_shapes);
  EXPECT_EQ(dtypes_and_shapes.value().size(), 2);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "0/one");
  EXPECT_EQ(dtypes_and_shapes.value()[1].name, "1/two");
}

TEST(FlatSignatureFromStructuredValueTest, TupleNaming) {
  tensorflow::StructuredValue value;
  *value.mutable_tuple_value()->add_values() = MakeLeaf("one");
  *value.mutable_tuple_value()->add_values() = MakeLeaf("two");

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromStructuredValue(value, &dtypes_and_shapes);
  EXPECT_EQ(dtypes_and_shapes.value().size(), 2);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "0/one");
  EXPECT_EQ(dtypes_and_shapes.value()[1].name, "1/two");
}

TEST(FlatSignatureFromStructuredValueTest, DictNaming) {
  tensorflow::StructuredValue value;
  (*value.mutable_dict_value()->mutable_fields())["a"] = MakeLeaf("one");
  (*value.mutable_dict_value()->mutable_fields())["b"] = MakeLeaf("two");

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromStructuredValue(value, &dtypes_and_shapes);
  EXPECT_EQ(dtypes_and_shapes.value().size(), 2);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "a/one");
  EXPECT_EQ(dtypes_and_shapes.value()[1].name, "b/two");
}

TEST(FlatSignatureFromStructuredValueTest, NamedTupleNaming) {
  tensorflow::StructuredValue value;
  value.mutable_named_tuple_value()->set_name("namedtuple");
  auto* one = value.mutable_named_tuple_value()->add_values();
  one->set_key("a");
  *one->mutable_value() = MakeLeaf("one");
  auto* two = value.mutable_named_tuple_value()->add_values();
  two->set_key("b");
  *two->mutable_value() = MakeLeaf("two");
  auto* three = value.mutable_named_tuple_value()->add_values();
  three->set_key("c");
  *three->mutable_value() = MakeLeaf("three");

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromStructuredValue(value, &dtypes_and_shapes);
  EXPECT_EQ(dtypes_and_shapes.value().size(), 3);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "a/one");
  EXPECT_EQ(dtypes_and_shapes.value()[1].name, "b/two");
  EXPECT_EQ(dtypes_and_shapes.value()[2].name, "c/three");
}

TEST(FlatSignatureFromStructuredValueTest, NestedNaming) {
  tensorflow::StructuredValue value;
  value.mutable_named_tuple_value()->set_name("namedtuple");
  auto* one = value.mutable_named_tuple_value()->add_values();
  one->set_key("a");
  *one->mutable_value()->mutable_list_value()->add_values() = MakeLeaf("one");
  *one->mutable_value()->mutable_list_value()->add_values() = MakeLeaf("two");
  auto* two = value.mutable_named_tuple_value()->add_values();
  two->set_key("b");
  *two->mutable_value() = MakeLeaf("three");
  auto* three = value.mutable_named_tuple_value()->add_values();
  three->set_key("c");
  *three->mutable_value() = MakeLeaf("four");

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromStructuredValue(value, &dtypes_and_shapes);
  EXPECT_EQ(dtypes_and_shapes.value().size(), 4);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "a/0/one");
  EXPECT_EQ(dtypes_and_shapes.value()[1].name, "a/1/two");
  EXPECT_EQ(dtypes_and_shapes.value()[2].name, "b/three");
  EXPECT_EQ(dtypes_and_shapes.value()[3].name, "c/four");
}

TEST(FlatSignatureFromStructuredValueTest, EmptyLeaf) {
  tensorflow::StructuredValue value;
  value.mutable_named_tuple_value()->set_name("namedtuple");
  auto* one = value.mutable_named_tuple_value()->add_values();
  one->set_key("a");
  *one->mutable_value()->mutable_list_value()->add_values() = MakeLeaf("one");
  *one->mutable_value()->mutable_list_value()->add_values() = MakeLeaf("");
  auto* two = value.mutable_named_tuple_value()->add_values();
  two->set_key("b");
  *two->mutable_value() = MakeLeaf("two");

  DtypesAndShapes dtypes_and_shapes = DtypesAndShapes::value_type({});
  auto status = FlatSignatureFromStructuredValue(value, &dtypes_and_shapes);
  EXPECT_EQ(dtypes_and_shapes.value().size(), 3);
  EXPECT_EQ(dtypes_and_shapes.value()[0].name, "a/0/one");
  EXPECT_EQ(dtypes_and_shapes.value()[1].name, "a/1");
  EXPECT_EQ(dtypes_and_shapes.value()[2].name, "b/two");
}

TEST(AddBatchDim, EmptyStructure) {
  tensorflow::StructuredValue value;
  REVERB_EXPECT_OK(AddBatchDim(&value, 10));
  EXPECT_THAT(value, EqualsProto(""));
}

TEST(AddBatchDim, NestedStructure) {
  auto value = testing::CreateProto<tensorflow::StructuredValue>(R"pb(
    dict_value {
      fields {
        key: "a"
        value {
          list_value {
            values {
              tensor_spec_value {
                name: "spec_1"
                shape {
                  dim { size: 5 }
                }
                dtype: DT_FLOAT
              }
            }
            values {
              bounded_tensor_spec_value {
                name: "bounded_spec_1"
                shape {}
                dtype: DT_INT32
                minimum {
                  dtype: DT_INT32
                  tensor_shape {}
                  int_val: 1
                }
                maximum {
                  dtype: DT_INT32
                  tensor_shape {}
                  int_val: 3
                }
              }
            }
          }
        }
      }
      fields {
        key: "b"
        value {
          tuple_value {
            values {
              tensor_spec_value {
                name: "spec_2"
                shape {
                  dim { size: 1 }
                }
                dtype: DT_DOUBLE
              }
            }
            values {
              named_tuple_value {
                name: "named_tuple"
                values {
                  key: "first"
                  value {
                    tensor_spec_value {
                      name: "spec_3"
                      shape {}
                      dtype: DT_BFLOAT16
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  )pb");
  REVERB_EXPECT_OK(AddBatchDim(&value, 10));
  EXPECT_THAT(value, EqualsProto(R"pb(
    dict_value {
      fields {
        key: "a"
        value {
          list_value {
            values {
              tensor_spec_value {
                name: "spec_1"
                shape {
                   dim { size: 10 }
                   dim { size: 5 }
                }
                dtype: DT_FLOAT
              }
            }
            values {
              bounded_tensor_spec_value {
                name: "bounded_spec_1"
                shape {
                  dim { size: 10 }
                }
                dtype: DT_INT32
                minimum {
                  dtype: DT_INT32
                  tensor_shape {}
                  int_val: 1
                }
                maximum {
                  dtype: DT_INT32
                  tensor_shape {}
                  int_val: 3
                }
              }
            }
          }
        }
      }
      fields {
        key: "b"
        value {
          tuple_value {
            values {
              tensor_spec_value {
                name: "spec_2"
                shape {
                  dim { size: 10 }
                  dim { size: 1 }
                }
                dtype: DT_DOUBLE
              }
            }
            values {
              named_tuple_value {
                name: "named_tuple"
                values {
                  key: "first"
                  value {
                    tensor_spec_value {
                      name: "spec_3"
                      shape {
                        dim { size: 10 }
                      }
                      dtype: DT_BFLOAT16
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  )pb"));
}

}  // namespace
}  // namespace internal
}  // namespace reverb
}  // namespace deepmind
