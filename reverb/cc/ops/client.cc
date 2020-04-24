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

#include <memory>
#include <string>
#include <vector>

#include <cstdint>
#include "absl/strings/str_cat.h"
#include "reverb/cc/replay_client.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {
namespace {

using ::tensorflow::tstring;
using ::tensorflow::errors::InvalidArgument;

REGISTER_OP("ReverbClient")
    .Output("handle: resource")
    .Attr("server_address: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Constructs a `ClientResource` that constructs a `ReplayClient` connected to
`server_address`. The resource allows ops to share the stub across calls.
)doc");

REGISTER_OP("ReverbClientSample")
    .Attr("Toutput_list: list(type) >= 0")
    .Input("handle: resource")
    .Input("table: string")
    .Output("key: uint64")
    .Output("probability: double")
    .Output("table_size: int64")
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

class ClientResource : public tensorflow::ResourceBase {
 public:
  explicit ClientResource(const std::string& server_address)
      : tensorflow::ResourceBase(),
        client_(server_address),
        server_address_(server_address) {}

  std::string DebugString() const override {
    return tensorflow::strings::StrCat("Client with server address: ",
                                       server_address_);
  }

  ReplayClient* client() { return &client_; }

 private:
  ReplayClient client_;
  std::string server_address_;

  TF_DISALLOW_COPY_AND_ASSIGN(ClientResource);
};

class ClientHandleOp : public tensorflow::ResourceOpKernel<ClientResource> {
 public:
  explicit ClientHandleOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::ResourceOpKernel<ClientResource>(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("server_address", &server_address_));
  }

 private:
  tensorflow::Status CreateResource(ClientResource** ret) override {
    *ret = new ClientResource(server_address_);
    return tensorflow::Status::OK();
  }

  std::string server_address_;

  TF_DISALLOW_COPY_AND_ASSIGN(ClientHandleOp);
};

// TODO(b/154929314): Change this to an async op.
class SampleOp : public tensorflow::OpKernel {
 public:
  explicit SampleOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    ClientResource* resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &resource));

    const tensorflow::Tensor* table_tensor;
    OP_REQUIRES_OK(context, context->input("table", &table_tensor));
    std::string table = table_tensor->scalar<tstring>()();

    std::vector<tensorflow::Tensor> sample;
    std::unique_ptr<ReplaySampler> sampler;

    ReplaySampler::Options options;
    options.max_samples = 1;
    options.max_in_flight_samples_per_worker = 1;

    OP_REQUIRES_OK(context,
                   resource->client()->NewSampler(table, options, &sampler));
    OP_REQUIRES_OK(context, sampler->GetNextTimestep(&sample, nullptr));
    OP_REQUIRES(context, sample.size() == context->num_outputs(),
                InvalidArgument(
                    "Number of tensors in the replay sample did not match the "
                    "expected count."));

    for (int i = 0; i < sample.size(); i++) {
      tensorflow::Tensor* tensor;
      OP_REQUIRES_OK(context,
                     context->allocate_output(i, sample[i].shape(), &tensor));
      *tensor = std::move(sample[i]);
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(SampleOp);
};

class UpdatePrioritiesOp : public tensorflow::OpKernel {
 public:
  explicit UpdatePrioritiesOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    ClientResource* resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &resource));

    const tensorflow::Tensor* table;
    OP_REQUIRES_OK(context, context->input("table", &table));
    const tensorflow::Tensor* keys;
    OP_REQUIRES_OK(context, context->input("keys", &keys));
    const tensorflow::Tensor* priorities;
    OP_REQUIRES_OK(context, context->input("priorities", &priorities));

    OP_REQUIRES(
        context, keys->dims() == 1,
        InvalidArgument("Tensors `keys` and `priorities` must be of rank 1."));
    OP_REQUIRES(context, keys->shape() == priorities->shape(),
                InvalidArgument(
                    "Tensors `keys` and `priorities` do not match in shape."));

    std::string table_str = table->scalar<tstring>()();
    std::vector<KeyWithPriority> updates;
    for (int i = 0; i < keys->dim_size(0); i++) {
      KeyWithPriority update;
      update.set_key(keys->flat<tensorflow::uint64>()(i));
      update.set_priority(priorities->flat<double>()(i));
      updates.push_back(std::move(update));
    }

    // The call will only fail if the Reverb-server is brought down during an
    // active call (e.g preempted). When this happens the request is retried and
    // since MutatePriorities sets `wait_for_ready` the request will no be sent
    // before the server is brought up again. It is therefore no problem to have
    // this retry in this tight loop.
    tensorflow::Status status;
    do {
      status = resource->client()->MutatePriorities(table_str, updates, {});
    } while (tensorflow::errors::IsUnavailable(status) ||
             tensorflow::errors::IsDeadlineExceeded(status));
    OP_REQUIRES_OK(context, status);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(UpdatePrioritiesOp);
};

class InsertOp : public tensorflow::OpKernel {
 public:
  explicit InsertOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    ClientResource* resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &resource));

    const tensorflow::Tensor* tables;
    OP_REQUIRES_OK(context, context->input("tables", &tables));
    const tensorflow::Tensor* priorities;
    OP_REQUIRES_OK(context, context->input("priorities", &priorities));

    OP_REQUIRES(context, tables->dims() == 1 && priorities->dims() == 1,
                InvalidArgument(
                    "Tensors `tables` and `priorities` must be of rank 1."));
    OP_REQUIRES(
        context, tables->shape() == priorities->shape(),
        InvalidArgument(
            "Tensors `tables` and `priorities` do not match in shape."));

    tensorflow::OpInputList data;
    OP_REQUIRES_OK(context, context->input_list("data", &data));

    // TODO(b/154929210): This can probably be avoided.
    std::vector<tensorflow::Tensor> tensors;
    for (const auto& i : data) {
      tensors.push_back(i);
    }

    std::unique_ptr<ReplayWriter> writer;
    OP_REQUIRES_OK(context,
                   resource->client()->NewWriter(1, 1, false, &writer));
    OP_REQUIRES_OK(context, writer->AppendTimestep(std::move(tensors)));

    auto tables_t = tables->flat<tstring>();
    auto priorities_t = priorities->flat<double>();
    for (int i = 0; i < tables->dim_size(0); i++) {
      OP_REQUIRES_OK(context,
                     writer->AddPriority(tables_t(i), 1, priorities_t(i)));
    }

    OP_REQUIRES_OK(context, writer->Close());
  }

  TF_DISALLOW_COPY_AND_ASSIGN(InsertOp);
};

REGISTER_KERNEL_BUILDER(Name("ReverbClient").Device(tensorflow::DEVICE_CPU),
                        ClientHandleOp);

REGISTER_KERNEL_BUILDER(
    Name("ReverbClientInsert").Device(tensorflow::DEVICE_CPU), InsertOp);

REGISTER_KERNEL_BUILDER(
    Name("ReverbClientSample").Device(tensorflow::DEVICE_CPU), SampleOp);

REGISTER_KERNEL_BUILDER(
    Name("ReverbClientUpdatePriorities").Device(tensorflow::DEVICE_CPU),
    UpdatePrioritiesOp);

}  // namespace
}  // namespace reverb
}  // namespace deepmind
