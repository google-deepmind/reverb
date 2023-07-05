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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace deepmind {
namespace reverb {
namespace {

REGISTER_OP("ReverbPatternDataset")
    .Input("dataset: variant")
    .Input("other_arguments: Targuments")
    .Attr("clear_after_episode: bool")
    .Attr("configs: list(string)")
    .Attr("is_end_of_episode: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("dtypes: list(type) >= 1")
    .Attr("shapes: list(shape) >= 1")
    .Attr("metadata: string = ''")
    .Output("output_dataset: variant")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Converts a dataset of steps into a dataset of trajectories by applying the
Reverb patterns (`configs`).

`clear_after_episode` indicates trajectories should respect episode boundaries
(if False, it may create trajectories that contain data from different
episodes).

`configs` is the list of reverb patterns to apply to the dataset of steps. The
 patterns are serialized StructuredWriterConfig protos.

`is_end_of_episode` is a function to apply to a step to indicate if this is the
 last one of an episode.

`dtypes` and `shapes` must match the types and shapes of the elements of the
output dataset.

`other_arguments` and `Targuments` refer to the other arguments of
`is_end_of_episode` and their types respectively.

This operation only supports eager mode (there is no TF1 support).
)doc");

}  // namespace
}  // namespace reverb
}  // namespace deepmind
