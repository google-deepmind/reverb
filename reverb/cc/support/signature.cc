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

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/trajectory_util.h"
#include "reverb/cc/table.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace deepmind {
namespace reverb {
namespace internal {

absl::Status FlatSignatureFromTableInfo(
    const TableInfo& info, DtypesAndShapes* dtypes_and_shapes) {
  if (!info.has_signature()) {
    *dtypes_and_shapes = absl::nullopt;
  } else {
    const auto& sig = info.signature();
    *dtypes_and_shapes = DtypesAndShapes::value_type{};
    auto status = FlatSignatureFromStructuredValue(sig, dtypes_and_shapes);
    if (!status.ok()) {
      return absl::Status(
          status.code(),
          absl::StrCat(status.message(), "Full signature struct: '",
                       info.signature().DebugString(), "'"));
    }
  }
  return absl::OkStatus();
}

namespace {

template <typename T>
std::string ExtendContext(absl::string_view context, T val) {
  const std::string val_string = absl::StrCat(val);
  return absl::StrCat(context,
                      ((!context.empty() && !val_string.empty()) ? "/" : ""),
                      val_string);
}

absl::Status FlatSignatureFromStructuredValue(
    const tensorflow::StructuredValue& value, absl::string_view context,
    DtypesAndShapes* dtypes_and_shapes) {
  switch (value.kind_case()) {
    case tensorflow::StructuredValue::kTensorSpecValue: {
      const auto& tensor_spec = value.tensor_spec_value();
      (*dtypes_and_shapes)
          ->push_back({ExtendContext(context, tensor_spec.name()),
                       tensor_spec.dtype(),
                       tensorflow::PartialTensorShape(tensor_spec.shape())});
    } break;
    case tensorflow::StructuredValue::kBoundedTensorSpecValue: {
      const auto& bounded_tensor_spec = value.bounded_tensor_spec_value();
      // This stores the dtype and shape of the boundary tensor spec. Currently,
      // the signature of such tensors only checked against these properties.
      // TODO(b/158033101): Make the signature check fully support boundaries.
      (*dtypes_and_shapes)
          ->push_back(
              {ExtendContext(context, bounded_tensor_spec.name()),
               bounded_tensor_spec.dtype(),
               tensorflow::PartialTensorShape(bounded_tensor_spec.shape())});
    } break;
    case tensorflow::StructuredValue::kListValue: {
      const auto& values = value.list_value().values();
      for (size_t i = 0; i < values.size(); ++i) {
        REVERB_RETURN_IF_ERROR(FlatSignatureFromStructuredValue(
            values[i], ExtendContext(context, i), dtypes_and_shapes));
      }
    } break;
    case tensorflow::StructuredValue::kTupleValue: {
      const auto& values = value.tuple_value().values();
      for (size_t i = 0; i < values.size(); ++i) {
        REVERB_RETURN_IF_ERROR(FlatSignatureFromStructuredValue(
            values[i], ExtendContext(context, i), dtypes_and_shapes));
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
        REVERB_RETURN_IF_ERROR(FlatSignatureFromStructuredValue(
            value.dict_value().fields().at(k), ExtendContext(context, k),
            dtypes_and_shapes));
      }
    } break;
    case tensorflow::StructuredValue::kNamedTupleValue: {
      for (const auto& p : value.named_tuple_value().values()) {
        REVERB_RETURN_IF_ERROR(FlatSignatureFromStructuredValue(
            p.value(), ExtendContext(context, p.key()), dtypes_and_shapes));
      }
    } break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Saw unsupported encoded subtree in signature: '",
                       value.DebugString(), "'"));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status FlatSignatureFromStructuredValue(
    const tensorflow::StructuredValue& value,
    DtypesAndShapes* dtypes_and_shapes) {
  return FlatSignatureFromStructuredValue(value, "", dtypes_and_shapes);
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
  for (int i = 0; i < chunk_data.data().tensors_size(); i++) {
    const auto& chunk = chunk_data.data().tensors(i);
    tensorflow::PartialTensorShape shape(chunk.tensor_shape());
    shape.RemoveDim(0);

    auto* spec =
        value.mutable_list_value()->add_values()->mutable_tensor_spec_value();
    spec->set_dtype(chunk.dtype());
    shape.AsProto(spec->mutable_shape());
  }

  return value;
}

tensorflow::StructuredValue StructuredValueFromItem(const TableItem& item) {
  tensorflow::StructuredValue value;

  auto get_tensor = [&](const FlatTrajectory::ChunkSlice& slice) {
    for (const auto& chunk : item.chunks) {
      if (chunk->key() == slice.chunk_key()) {
        return &chunk->data().data().tensors(slice.index());
      }
    }
    REVERB_CHECK(false) << "Invalid item.";
  };

  for (int col_idx = 0; col_idx < item.item.flat_trajectory().columns_size();
       col_idx++) {
    const auto& col = item.item.flat_trajectory().columns(col_idx);
    const auto* tensor_proto = get_tensor(col.chunk_slices(0));

    auto* spec =
        value.mutable_list_value()->add_values()->mutable_tensor_spec_value();
    spec->set_dtype(tensor_proto->dtype());
    *spec->mutable_shape() = tensor_proto->tensor_shape();

    if (col.squeeze()) {
      spec->mutable_shape()->mutable_dim()->DeleteSubrange(0, 1);
    } else {
      spec->mutable_shape()->mutable_dim(0)->set_size(-1);
    }
  }

  return value;
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
