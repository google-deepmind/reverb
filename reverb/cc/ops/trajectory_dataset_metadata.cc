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

REGISTER_OP("ReverbTrajectoryDataset")
    .Input("server_address: string")
    .Input("table: string")
    .Attr("max_in_flight_samples_per_worker: int = 100")
    .Attr("num_workers_per_iterator: int = -1")
    .Attr("max_samples_per_stream: int = -1")
    .Attr("rate_limiter_timeout_ms: int = -1")
    .Attr("max_samples: int = -1")
    .Attr("dtypes: list(type) >= 1")
    .Attr("shapes: list(shape) >= 1")
    .Output("dataset: variant")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Establishes and manages a connection to gRPC ReverbService at `server_address`
to stream samples from table `table`.

The connection is managed using a single instance of `Client` (see
../client.h) owned by the Dataset. From the shared `Client`, each iterator
maintains their own `Sampler` (see ../sampler.h), allowing for multiple
parallel streams using a single connection.

`dtypes` and `shapes` must match the type and shape of the trajectories
referenced by items in `table`.

`max_in_flight_samples_per_worker` (defaults to 100) is the maximum number of
 sampled item allowed to exist in flight (per iterator). See
`Sampler::Options::max_in_flight_samples_per_worker` for more details.

`num_workers_per_iterator` (defaults to -1, i.e. auto selected) is the number of
worker threads to start per iterator. When the selected table uses a FIFO
sampler (i.e. a queue) then exactly 1 worker must be used to avoid races causing
invalid ordering of items. For all other samplers, this value should be roughly
equal to the number of threads available on the CPU.

`max_samples_per_stream` (defaults to -1, i.e. auto selected) is the maximum
number of samples to fetch from a stream before a new call is made. Keeping this
number low ensures that the data is fetched uniformly from all servers.

`rate_limiter_timeout_ms` (defaults to -1, i.e. never time out) is the number of
milliseconds an iterator should wait for new data from the sampler before timing
out. This can be useful, e.g., when the Reverb server receives data in
collection stages - and a dataset iterator should stop when no new data is
available for a while. If `rate_limiter_timeout_ms >= 0`, an iterator that waits
for data longer than this will close and mark the input sequence as finished.
Note that the timeout behavior depends on the Table's rate limiter. For example,
the table may contain data, but the rate limiter may pause sampling - and this
can cause a timeout to occur. Note also that when `num_workers_per_iterator >
1`, a timeout on any given worker will cause a timeout for the dataset.

`max_samples` (defaults to -1, i.e. infinite) is the maximum number of samples
to fetch from the server. Once `max_samples` samples have been returned the
iterator will close. This can be used when it is necessary to fetch an exact
number of items (thus avoiding the prefetching that otherwise is implemented by
tensorflow).
)doc");

}  // namespace
}  // namespace reverb
}  // namespace deepmind
