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

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace deepmind {
namespace reverb {
namespace internal {

tensorflow::Status FlatSignatureFromTableInfo(
    const TableInfo& info, DtypesAndShapes* dtypes_and_shapes) {
  if (!info.has_signature()) {
    *dtypes_and_shapes = absl::nullopt;
  } else {
    const auto& sig = info.signature();
    *dtypes_and_shapes = DtypesAndShapes::value_type{};
    auto status = FlatSignatureFromStructuredValue(sig, dtypes_and_shapes);
    if (!status.ok()) {
      tensorflow::errors::AppendToMessage(&status, "Full signature struct: '",
                                          info.signature().DebugString(), "'");
      return status;
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status FlatSignatureFromStructuredValue(
    const tensorflow::StructuredValue& value,
    DtypesAndShapes* dtypes_and_shapes) {
  switch (value.kind_case()) {
    case tensorflow::StructuredValue::kTensorSpecValue: {
      const auto& tensor_spec = value.tensor_spec_value();
      (*dtypes_and_shapes)
          ->push_back({tensor_spec.name(), tensor_spec.dtype(),
                       tensorflow::PartialTensorShape(tensor_spec.shape())});
    } break;
    case tensorflow::StructuredValue::kBoundedTensorSpecValue: {
      const auto& bounded_tensor_spec = value.bounded_tensor_spec_value();
      // This stores the dtype and shape of the boundary tensor spec. Currently,
      // the signature of such tensors only checked against these properties.
      // TODO(b/158033101): Make the signature check fully support boundaries.
      (*dtypes_and_shapes)
          ->push_back(
              {bounded_tensor_spec.name(), bounded_tensor_spec.dtype(),
               tensorflow::PartialTensorShape(bounded_tensor_spec.shape())});
    } break;
    case tensorflow::StructuredValue::kListValue: {
      for (const auto& v : value.list_value().values()) {
        TF_RETURN_IF_ERROR(
            FlatSignatureFromStructuredValue(v, dtypes_and_shapes));
      }
    } break;
    case tensorflow::StructuredValue::kTupleValue: {
      for (const auto& v : value.tuple_value().values()) {
        TF_RETURN_IF_ERROR(
            FlatSignatureFromStructuredValue(v, dtypes_and_shapes));
      }
    } break;
    case tensorflow::StructuredValue::kDictValue: {
      std::vector<std::string> keys;
      keys.reserve(value.dict_value().fields_size());
      for (const auto& f : value.dict_value().fields()) {
        keys.push_back(f.first);
      }
      std::sort(keys.begin(), keys.end());
      for (const auto& k : keys) {
        TF_RETURN_IF_ERROR(FlatSignatureFromStructuredValue(
            value.dict_value().fields().at(k), dtypes_and_shapes));
      }
    } break;
    case tensorflow::StructuredValue::kNamedTupleValue: {
      for (const auto &p : value.named_tuple_value().values()) {
        TF_RETURN_IF_ERROR(FlatSignatureFromStructuredValue(
            p.value(), dtypes_and_shapes));
      }
    } break;
    default:
      return tensorflow::errors::InvalidArgument(
          "Saw unsupported encoded subtree in signature: '",
          value.DebugString(), "'");
  }
  return tensorflow::Status::OK();
}

std::string DtypesShapesString(
    const std::vector<internal::TensorSpec>& dtypes_and_shapes) {
  std::vector<std::string> strings;
  strings.reserve(dtypes_and_shapes.size());
  for (int i = 0; i < dtypes_and_shapes.size(); ++i) {
    const auto& p = dtypes_and_shapes[i];
    strings.push_back(absl::StrCat(i, ": Tensor<name: '", p.name, "', dtype: ",
                                   tensorflow::DataTypeString(p.dtype),
                                   ", shape: ", p.shape.DebugString(), ">"));
  }
  return absl::StrJoin(strings, ", ");
}

std::string DtypesShapesString(const std::vector<tensorflow::Tensor>& tensors) {
  return DtypesShapesString(SpecsFromTensors(tensors));
}

tensorflow::StructuredValue StructuredValueFromChunkData(
    const ChunkData& chunk_data) {
  tensorflow::StructuredValue value;
  for (int i = 0; i < chunk_data.data_size(); i++) {
    const auto& chunk = chunk_data.data(i);
    tensorflow::PartialTensorShape shape(chunk.tensor_shape());
    shape.RemoveDim(0);

    auto* spec =
        value.mutable_list_value()->add_values()->mutable_tensor_spec_value();
    spec->set_dtype(chunk.dtype());
    shape.AsProto(spec->mutable_shape());
  }

  return value;
}

tensorflow::Status FlatPathFromStructuredValue(
    const tensorflow::StructuredValue& value, absl::string_view prefix,
    std::vector<std::string>* paths) {
  switch (value.kind_case()) {
    case tensorflow::StructuredValue::kTensorSpecValue:
      paths->push_back(std::string(prefix));
      break;
    case tensorflow::StructuredValue::kBoundedTensorSpecValue:
      // The path does not store the bounds from the bounded spec as currently
      // those bounds are not checked as part of the signature check.
      // TODO(b/158033101): Make the signature check fully support boundaries.
      paths->push_back(std::string(prefix));
      break;
    case tensorflow::StructuredValue::kListValue: {
      for (int i = 0; i < value.list_value().values_size(); i++) {
        TF_RETURN_IF_ERROR(FlatPathFromStructuredValue(
            value.list_value().values(i), absl::StrCat(prefix, "[", i, "]"),
            paths));
      }
    } break;
    case tensorflow::StructuredValue::kTupleValue: {
      for (int i = 0; i < value.tuple_value().values_size(); i++) {
        TF_RETURN_IF_ERROR(FlatPathFromStructuredValue(
            value.tuple_value().values(i), absl::StrCat(prefix, "[", i, "]"),
            paths));
      }
    } break;
    case tensorflow::StructuredValue::kDictValue: {
      std::vector<std::string> keys;
      keys.reserve(value.dict_value().fields_size());
      for (const auto& f : value.dict_value().fields()) {
        keys.push_back(f.first);
      }
      std::sort(keys.begin(), keys.end());
      for (const auto& k : keys) {
        TF_RETURN_IF_ERROR(
            FlatPathFromStructuredValue(value.dict_value().fields().at(k),
                                        absl::StrCat(prefix, ".", k), paths));
      }
    } break;
    case tensorflow::StructuredValue::kNamedTupleValue: {
      for (const auto& p : value.named_tuple_value().values()) {
        TF_RETURN_IF_ERROR(FlatPathFromStructuredValue(
            p.value(), absl::StrCat(prefix, ".", p.key()), paths));
      }
    } break;
    default:
      return tensorflow::errors::InvalidArgument(
          "Saw unsupported encoded subtree in signature: '",
          value.DebugString(), "'");
  }
  return tensorflow::Status::OK();
}

std::vector<internal::TensorSpec> SpecsFromTensors(
    const std::vector<tensorflow::Tensor>& tensors) {
  std::vector<internal::TensorSpec> spec;
  spec.reserve(tensors.size());
  for (const auto& t : tensors) {
    spec.push_back({"", t.dtype(), t.shape()});
  }
  return spec;
}


}  // namespace internal
}  // namespace reverb
}  // namespace deepmind
