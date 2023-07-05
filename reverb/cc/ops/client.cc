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

#include "reverb/cc/client.h"

#include <memory>
#include <string>
#include <vector>

#include <cstdint>
#include "absl/strings/str_cat.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/support/tf_util.h"
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

class ClientResource : public tensorflow::ResourceBase {
 public:
  explicit ClientResource(const std::string& server_address)
      : tensorflow::ResourceBase(),
        client_(server_address),
        server_address_(server_address) {}

  std::string DebugString() const override {
    return absl::StrCat("Client with server address: ", server_address_);
  }

  Client* client() { return &client_; }

 private:
  Client client_;
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
    return tensorflow::OkStatus();
  }

  std::string server_address_;

  TF_DISALLOW_COPY_AND_ASSIGN(ClientHandleOp);
};

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

    std::unique_ptr<Sampler> sampler;

    Sampler::Options options;
    options.max_samples = 1;
    options.max_in_flight_samples_per_worker = 1;

    constexpr auto kValidationTimeout = absl::Seconds(30);
    OP_REQUIRES_OK(
        context, ToTensorflowStatus(resource->client()->NewSampler(
                     table, options, /*validation_timeout=*/kValidationTimeout,
                     &sampler)));

    std::vector<tensorflow::Tensor> data;
    std::shared_ptr<const SampleInfo> info;
    OP_REQUIRES_OK(context, ToTensorflowStatus(sampler->GetNextTimestep(
                                &data, nullptr, &info)));
    OP_REQUIRES(
        context,
        data.size() + Sampler::kNumInfoTensors == context->num_outputs(),
        InvalidArgument(
            "Number of tensors in the replay sample did not match the "
            "expected count. Got ",
            data.size() + Sampler::kNumInfoTensors, " but wanted ",
            context->num_outputs()));

    auto flat_sample = Sampler::WithInfoTensors(*info, std::move(data));
    for (int i = 0; i < flat_sample.size(); i++) {
      tensorflow::Tensor* tensor;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  i, flat_sample[i].shape(), &tensor));
      *tensor = std::move(flat_sample[i]);
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
                    "Tensors `keys` and `priorities` do not match in shape (",
                    keys->shape().DebugString(), " vs. ",
                    priorities->shape().DebugString(), ")"));

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
    absl::Status status;
    do {
      status = resource->client()->MutatePriorities(table_str, updates, {});
    } while (absl::IsUnavailable(status) || absl::IsDeadlineExceeded(status));
    OP_REQUIRES_OK(context, ToTensorflowStatus(status));
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

    std::vector<tensorflow::Tensor> tensors;
    for (const auto& i : data) {
      tensors.push_back(i);
    }

    std::unique_ptr<Writer> writer;
    OP_REQUIRES_OK(context, ToTensorflowStatus(resource->client()->NewWriter(
                                1, 1, false, &writer)));
    OP_REQUIRES_OK(context,
                   ToTensorflowStatus(writer->Append(std::move(tensors))));

    auto tables_t = tables->flat<tstring>();
    auto priorities_t = priorities->flat<double>();
    for (int i = 0; i < tables->dim_size(0); i++) {
      OP_REQUIRES_OK(context, ToTensorflowStatus(writer->CreateItem(
                                  tables_t(i), 1, priorities_t(i))));
    }

    OP_REQUIRES_OK(context, ToTensorflowStatus(writer->Flush()));

    OP_REQUIRES_OK(context, ToTensorflowStatus(writer->Close()));
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
