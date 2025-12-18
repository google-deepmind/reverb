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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "reverb/cc/client.h"
#include "reverb/cc/errors.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/sampler.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/macros.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/tstring.h"

namespace deepmind {
namespace reverb {
namespace {

using ::tensorflow::errors::Cancelled;
using ::tensorflow::errors::FailedPrecondition;
using ::tensorflow::errors::Unimplemented;

class ReverbTimestepDatasetOp : public tensorflow::data::DatasetOpKernel {
 public:
  explicit ReverbTimestepDatasetOp(tensorflow::OpKernelConstruction* ctx)
      : tensorflow::data::DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("max_in_flight_samples_per_worker",
                          &sampler_options_.max_in_flight_samples_per_worker));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_workers_per_iterator",
                                     &sampler_options_.num_workers));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_samples_per_stream",
                                     &sampler_options_.max_samples_per_stream));
    int64_t rate_limiter_timeout_ms;
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("rate_limiter_timeout_ms", &rate_limiter_timeout_ms));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("max_samples", &sampler_options_.max_samples));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));

    sampler_options_.rate_limiter_timeout =
        Int64MillisToNonnegativeDuration(rate_limiter_timeout_ms);

    OP_REQUIRES_OK(ctx, sampler_options_.Validate());
  }

  void MakeDataset(tensorflow::OpKernelContext* ctx,
                   tensorflow::data::DatasetBase** output) override {
    tensorflow::tstring server_address;
    tensorflow::tstring table;
    OP_REQUIRES_OK(ctx,
                   tensorflow::data::ParseScalarArgument<tensorflow::tstring>(
                       ctx, "server_address", &server_address));
    OP_REQUIRES_OK(ctx,
                   tensorflow::data::ParseScalarArgument<tensorflow::tstring>(
                       ctx, "table", &table));

    *output = new Dataset(ctx, server_address, dtypes_, shapes_, table,
                          sampler_options_);
  }

 private:
  class Dataset : public tensorflow::data::DatasetBase {
   public:
    Dataset(tensorflow::OpKernelContext* ctx, std::string server_address,
            tensorflow::DataTypeVector dtypes,
            std::vector<tensorflow::PartialTensorShape> shapes,
            std::string table, const Sampler::Options& sampler_options)
        : tensorflow::data::DatasetBase(tensorflow::data::DatasetContext(ctx)),
          server_address_(std::move(server_address)),
          dtypes_(std::move(dtypes)),
          shapes_(std::move(shapes)),
          table_(std::move(table)),
          sampler_options_(sampler_options),
          client_(std::make_unique<Client>(server_address_)) {}

    std::unique_ptr<tensorflow::data::IteratorBase> MakeIteratorInternal(
        const std::string& prefix) const override {
      return std::make_unique<Iterator>(
          tensorflow::data::DatasetIterator<Dataset>::Params{
              this, absl::StrCat(prefix, "::ReverbTimestepDataset")},
          client_.get(), table_, sampler_options_, dtypes_, shapes_);
    }

    const tensorflow::DataTypeVector& output_dtypes() const override {
      return dtypes_;
    }

    const std::vector<tensorflow::PartialTensorShape>& output_shapes()
        const override {
      return shapes_;
    }

    std::string DebugString() const override {
      return "ReverbTimestepDatasetOp::Dataset";
    }

    absl::Status CheckExternalState() const override {
      return FailedPrecondition(DebugString(), " depends on external state.");
    }

    absl::Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
      inputs->clear();
      return absl::OkStatus();
    }

   protected:
    absl::Status AsGraphDefInternal(tensorflow::data::SerializationContext* ctx,
                                    DatasetGraphDefBuilder* b,
                                    tensorflow::Node** output) const override {
      tensorflow::AttrValue max_in_flight_samples_per_worker_attr;
      tensorflow::AttrValue num_workers_attr;
      tensorflow::AttrValue max_samples_per_stream_attr;
      tensorflow::AttrValue rate_limiter_timeout_ms_attr;
      tensorflow::AttrValue max_samples_attr;
      tensorflow::AttrValue dtypes_attr;
      tensorflow::AttrValue shapes_attr;

      tensorflow::Node* server_address = nullptr;
      tensorflow::Node* table = nullptr;
      TF_RETURN_IF_ERROR(
          b->AddScalar<tensorflow::tstring>(server_address_, &server_address));
      TF_RETURN_IF_ERROR(b->AddScalar<tensorflow::tstring>(table_, &table));

      b->BuildAttrValue(sampler_options_.max_in_flight_samples_per_worker,
                        &max_in_flight_samples_per_worker_attr);
      b->BuildAttrValue(sampler_options_.num_workers, &num_workers_attr);
      b->BuildAttrValue(sampler_options_.max_samples_per_stream,
                        &max_samples_per_stream_attr);
      b->BuildAttrValue(
          static_cast<int64_t>(NonnegativeDurationToInt64Millis(
              sampler_options_.rate_limiter_timeout)),
          &rate_limiter_timeout_ms_attr);
      b->BuildAttrValue(sampler_options_.max_samples, &max_samples_attr);
      b->BuildAttrValue(dtypes_, &dtypes_attr);
      b->BuildAttrValue(shapes_, &shapes_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this,
          /*inputs=*/{server_address, table},
          /*attrs=*/
          {
              {"max_in_flight_samples_per_worker",
               max_in_flight_samples_per_worker_attr},
              {"num_workers_per_iterator", num_workers_attr},
              {"max_samples_per_stream", max_samples_per_stream_attr},
              {"rate_limiter_timeout_ms", rate_limiter_timeout_ms_attr},
              {"max_samples", max_samples_attr},
              {"dtypes", dtypes_attr},
              {"shapes", shapes_attr},
          },
          output));

      return absl::OkStatus();
    }

   private:
    class Iterator : public tensorflow::data::DatasetIterator<Dataset> {
     public:
      explicit Iterator(
          const Params& params, Client* client, const std::string& table,
          const Sampler::Options& sampler_options,
          const tensorflow::DataTypeVector& dtypes,
          const std::vector<tensorflow::PartialTensorShape>& shapes)
          : DatasetIterator<Dataset>(params),
            client_(client),
            table_(table),
            sampler_options_(sampler_options),
            dtypes_(dtypes),
            shapes_(shapes),
            rate_limited_(false) {}

      absl::Status Initialize(tensorflow::data::IteratorContext* ctx) override {
        constexpr auto kValidationTimeout = absl::Seconds(30);

        // The shapes and dtypes contains metadata fields but the signature does
        // not.
        tensorflow::DataTypeVector validation_dtypes(
            dtypes_.begin() + Sampler::kNumInfoTensors, dtypes_.end());
        std::vector<tensorflow::PartialTensorShape> validation_shapes(
            shapes_.begin() + Sampler::kNumInfoTensors, shapes_.end());

        auto status = client_->NewSampler(table_, sampler_options_,
                                          validation_dtypes, validation_shapes,
                                          kValidationTimeout, &sampler_);
        if (absl::IsDeadlineExceeded(status)) {
          REVERB_LOG(REVERB_WARNING)
              << "Unable to validate shapes and dtypes of new sampler for '"
              << table_ << "' as server could not be reached in time ("
              << kValidationTimeout
              << "). We were thus unable to fetch signature from server. The "
                 "sampler will be constructed without validating the dtypes "
                 "and shapes.";
          // Ask for a NewSampler with negative validation_timeout Duration,
          // which causes it to skip the validation and return an OK status.
          return client_->NewSampler(
              table_, sampler_options_,
              /*validation_timeout=*/-absl::InfiniteDuration(), &sampler_);
        }
        return status;
      }

      absl::Status GetNextInternal(tensorflow::data::IteratorContext* ctx,
                                   std::vector<tensorflow::Tensor>* out_tensors,
                                   bool* end_of_sequence) override {
        REVERB_CHECK(sampler_.get() != nullptr) << "Initialize was not called?";

        auto token = ctx->cancellation_manager()->get_cancellation_token();
        bool registered = ctx->cancellation_manager()->RegisterCallback(
            token, [&] { sampler_->Close(); });
        if (!registered) {
          sampler_->Close();
        }

        std::vector<tensorflow::Tensor> data;
        std::shared_ptr<const SampleInfo> info;
        bool last_timestep = false;
        absl::Status status =
            sampler_->GetNextTimestep(&data, &last_timestep, &info);

        if (registered &&
            !ctx->cancellation_manager()->DeregisterCallback(token)) {
          return Cancelled("Iterator context was cancelled");
        }

        if (status.ok()) {
          *end_of_sequence = false;
          *out_tensors = Sampler::WithInfoTensors(*info, std::move(data));
          rate_limited_ = info->rate_limited();
          return absl::OkStatus();
        } else if (sampler_options_.rate_limiter_timeout <
                       absl::InfiniteDuration() &&
                   errors::IsRateLimiterTimeout(status)) {
          *end_of_sequence = true;
          return absl::OkStatus();
        } else if (absl::IsOutOfRange(status) &&
                   sampler_options_.max_samples > 0) {
          // `max_samples` samples have already been returned by the iterator.
          *end_of_sequence = true;
          return absl::OkStatus();
        } else {
          return status;
        }
      }

     protected:
      absl::Status SaveInternal(
          tensorflow::data::SerializationContext* ctx,
          tensorflow::data::IteratorStateWriter* writer) override {
        return Unimplemented("SaveInternal is currently not supported");
      }

      absl::Status RestoreInternal(
          tensorflow::data::IteratorContext* ctx,
          tensorflow::data::IteratorStateReader* reader) override {
        return Unimplemented("RestoreInternal is currently not supported");
      }

      tensorflow::data::TraceMeMetadata GetTraceMeMetadata() const override {
        return {{"rate_limited", rate_limited_ ? "true" : "false"}};
      }

     private:
      Client* client_;
      const std::string& table_;
      const Sampler::Options sampler_options_;
      const tensorflow::DataTypeVector& dtypes_;
      const std::vector<tensorflow::PartialTensorShape>& shapes_;
      std::unique_ptr<Sampler> sampler_;

      // Whether the active sample was delayed due to rate limiting.
      bool rate_limited_;
    };  // Iterator.

    const std::string server_address_;
    const tensorflow::DataTypeVector dtypes_;
    const std::vector<tensorflow::PartialTensorShape> shapes_;
    const std::string table_;
    const Sampler::Options sampler_options_;
    std::unique_ptr<Client> client_;
  };  // Dataset.

  Sampler::Options sampler_options_;
  tensorflow::DataTypeVector dtypes_;
  std::vector<tensorflow::PartialTensorShape> shapes_;

  TF_DISALLOW_COPY_AND_ASSIGN(ReverbTimestepDatasetOp);
};

REGISTER_KERNEL_BUILDER(
    Name("ReverbTimestepDataset").Device(tensorflow::DEVICE_CPU),
    ReverbTimestepDatasetOp);

}  // namespace
}  // namespace reverb
}  // namespace deepmind
