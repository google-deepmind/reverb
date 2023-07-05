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

REGISTER_OP("ReverbClient")
    .Output("handle: resource")
    .Attr("server_address: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Constructs a `ClientResource` that constructs a `Client` connected to
`server_address`. The resource allows ops to share the stub across calls.
)doc");

REGISTER_OP("ReverbClientSample")
    .Attr("Toutput_list: list(type) >= 0")
    .Input("handle: resource")
    .Input("table: string")
    .Output("key: uint64")
    .Output("probability: double")
    .Output("table_size: int64")
    .Output("priority: double")
    .Output("times_sampled: int32")
    .Output("outputs: Toutput_list")
    .Doc(R"doc(
Blocking call to sample a single item from table `table` using shared resource.
A `SampleStream`-stream is opened  between the client and the server and when
the one sample has been received, the stream is closed.

Prefer to use `ReverbDataset` when requesting more than one sample to avoid
opening and closing the stream with each call.
)doc");

REGISTER_OP("ReverbClientUpdatePriorities")
    .Input("handle: resource")
    .Input("table: string")
    .Input("keys: uint64")
    .Input("priorities: double")
    .Doc(R"doc(
Blocking call to update the priorities of a collection of items. Keys that could
not be found in table `table` on server are ignored and does not impact the rest
of the request.
)doc");

REGISTER_OP("ReverbClientInsert")
    .Attr("T: list(type) >= 0")
    .Input("handle: resource")
    .Input("data: T")
    .Input("tables: string")
    .Input("priorities: double")
    .Doc(R"doc(
Blocking call to insert a single trajectory into one or more tables. The data
is treated as an episode constituting of a single timestep. Note that this mean
that when the item is sampled, it will be returned as a sequence of length 1,
containing `data`.
)doc");

}  // namespace
}  // namespace reverb
}  // namespace deepmind
