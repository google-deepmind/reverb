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

#ifndef REVERB_CC_SUPPORT_SIGNATURE_H_
#define REVERB_CC_SUPPORT_SIGNATURE_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/schema.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace deepmind {
namespace reverb {
namespace internal {

struct TensorSpec {
  std::string name;
  tensorflow::DataType dtype;
  tensorflow::PartialTensorShape shape;
};

typedef absl::optional<std::vector<TensorSpec>> DtypesAndShapes;

absl::Status FlatSignatureFromTableInfo(
    const TableInfo& info, DtypesAndShapes* dtypes_and_shapes);

absl::Status FlatSignatureFromStructuredValue(
    const tensorflow::StructuredValue& value,
    DtypesAndShapes* dtypes_and_shapes);

tensorflow::StructuredValue StructuredValueFromChunkData(
    const ChunkData& chunk_data);

// Map from table name to optional vector of flattened (dtype, shape) pairs.
typedef internal::flat_hash_map<std::string, internal::DtypesAndShapes>
    FlatSignatureMap;

std::string DtypesShapesString(
    const std::vector<internal::TensorSpec>& dtypes_and_shapes);
std::string DtypesShapesString(const std::vector<tensorflow::Tensor>& tensors);

std::vector<internal::TensorSpec> SpecsFromTensors(
    const std::vector<tensorflow::Tensor>& tensors);

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_SIGNATURE_H_
