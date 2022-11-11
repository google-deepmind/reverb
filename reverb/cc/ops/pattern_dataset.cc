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

#include <deque>
#include <algorithm>
#include <functional>
#include <memory>
#include <queue>
#include <vector>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/ops/queue_writer.h"
#include "reverb/cc/patterns.pb.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/structured_writer.h"
#include "reverb/cc/support/tf_util.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/status.h"

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

class ReverbPatternDatasetOp : public tensorflow::data::UnaryDatasetOpKernel {
 public:
  explicit ReverbPatternDatasetOp(tensorflow::OpKernelConstruction* ctx)
      : tensorflow::data::UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(
        ctx, tensorflow::data::FunctionMetadata::Create(
                 ctx, "is_end_of_episode", /*params=*/{}, &func_metadata_));
    OP_REQUIRES(
        ctx, func_metadata_->short_circuit_info().indices.size() <= 1,
        tensorflow::errors::InvalidArgument(
            "is_end_of_episode function has more than one return value."));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("clear_after_episode", &clear_after_episode_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));
    std::vector<tensorflow::tstring> serialized_configs;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("configs", &serialized_configs));

    configs_.reserve(serialized_configs.size());
    required_keep_alive_ = 0;
    for (const auto& c : serialized_configs) {
      StructuredWriterConfig proto;
      OP_REQUIRES(ctx, proto.ParseFromString(c),
                  tensorflow::errors::InvalidArgument(
                      "`serialized_configs` could not be parsed as "
                      "`StructuredWriterConfig`. The first invalid config is: ",
                      c));

      // Find the history length required by this config.
      int history_length = 0;
      for (const auto& node : proto.flat()) {
        history_length = std::max(
            history_length, std::abs(std::min(node.start(), node.stop())));
      }

      // Update the global maximum history length required.
      required_keep_alive_ = std::max(required_keep_alive_, history_length);
      configs_.push_back(proto);
    }
  }

  void MakeDataset(tensorflow::OpKernelContext* ctx,
                   tensorflow::data::DatasetBase* input,
                   tensorflow::data::DatasetBase** output) override {
    std::unique_ptr<tensorflow::data::CapturedFunction> captured_func;

    OP_REQUIRES_OK(ctx,
                   tensorflow::data::CapturedFunction::Create(
                       ctx, func_metadata_, "other_arguments", &captured_func));

    *output = new Dataset(ctx, input, configs_,
                          required_keep_alive_, clear_after_episode_,
                          std::move(captured_func), dtypes_, shapes_);
  }

 private:
  class Dataset : public tensorflow::data::DatasetBase {
   public:
    Dataset(tensorflow::OpKernelContext* ctx,
            tensorflow::data::DatasetBase* input,
            std::vector<StructuredWriterConfig> configs,
            int required_keep_alive, bool clear_after_episode,
            std::unique_ptr<tensorflow::data::CapturedFunction> captured_func,
            tensorflow::DataTypeVector dtypes,
            std::vector<tensorflow::PartialTensorShape> shapes)
        : tensorflow::data::DatasetBase(tensorflow::data::DatasetContext(ctx)),
          input_(input),
          required_keep_alive_(required_keep_alive),
          configs_(std::move(configs)),
          clear_after_episode_(clear_after_episode),
          captured_func_(std::move(captured_func)),
          dtypes_(std::move(dtypes)),
          shapes_(std::move(shapes)) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<tensorflow::data::IteratorBase> MakeIteratorInternal(
        const std::string& prefix) const override {
      return std::make_unique<Iterator>(
          tensorflow::data::DatasetIterator<Dataset>::Params{
              this, absl::StrCat(prefix, "::ReverbPatternDataset")},
          configs_, required_keep_alive_, clear_after_episode_, dtypes_,
          shapes_);
    }

    const tensorflow::DataTypeVector& output_dtypes() const override {
      return dtypes_;
    }

    const std::vector<tensorflow::PartialTensorShape>& output_shapes()
        const override {
      return shapes_;
    }

    std::string DebugString() const override {
      return "ReverbPatternDatasetOp::Dataset";
    }

    int64_t CardinalityInternal() const override {
      int64_t n = input_->Cardinality();
      if (n == tensorflow::data::kInfiniteCardinality) {
        // We don't know what's the cardinality of the output as it depends on
        // the patterns and the conditions of each config. However, if the input
        // has kInfiniteCardinality, the output is "Infinite" as well.
        return n;
      }
      return tensorflow::data::kUnknownCardinality;
    }

    tensorflow::Status InputDatasets(
        std::vector<const tensorflow::data::DatasetBase*>* inputs)
        const override {
      inputs->push_back(input_);
      return tensorflow::OkStatus();
    }

    tensorflow::Status CheckExternalState() const override {
      TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
      return input_->CheckExternalState();
    }

   protected:
    tensorflow::Status AsGraphDefInternal(
        tensorflow::data::SerializationContext* ctx, DatasetGraphDefBuilder* b,
        tensorflow::Node** output) const override {
      tensorflow::AttrValue dtypes_attr;
      tensorflow::AttrValue shapes_attr;
      tensorflow::AttrValue configs_attr;
      tensorflow::AttrValue clear_after_episode_attr;
      tensorflow::Node* input_graph_node = nullptr;
      std::vector<tensorflow::Node*> other_arguments;
      tensorflow::DataTypeVector other_arguments_types;
      tensorflow::AttrValue func_attr;
      tensorflow::AttrValue other_arguments_types_attr;

      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

      TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                    &other_arguments_types));
      b->BuildAttrValue(captured_func_->func(), &func_attr);
      b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

      b->BuildAttrValue(clear_after_episode_, &clear_after_episode_attr);

      std::vector<tensorflow::tstring> serialized_configs;
      serialized_configs.reserve(configs_.size());
      for (const auto& c : configs_){
        serialized_configs.push_back(c.SerializeAsString());
      }

      b->BuildAttrValue(serialized_configs, &configs_attr);
      b->BuildAttrValue(dtypes_, &dtypes_attr);
      b->BuildAttrValue(shapes_, &shapes_attr);

      TF_RETURN_IF_ERROR(
          b->AddDataset(this,
                        // We have to separate single-node inputs from vector
                        // inputs but indicate the global order.
                        /*inputs=*/
                        {
                            {0, input_graph_node},
                        },
                        /*list_inputs=*/
                        {
                            {1, other_arguments},
                        },
                        /*attrs=*/
                        {
                            {"clear_after_episode", clear_after_episode_attr},
                            {"configs", configs_attr},
                            {"is_end_of_episode", func_attr},
                            {"Targuments", other_arguments_types_attr},
                            {"dtypes", dtypes_attr},
                            {"shapes", shapes_attr},
                        },
                        output));

      return tensorflow::OkStatus();
    }

   private:
    class Iterator : public tensorflow::data::DatasetIterator<Dataset> {
     public:
      explicit Iterator(
          const Params& params, std::vector<StructuredWriterConfig> configs,
          int required_keep_alive, bool clear_after_episode,
          const tensorflow::DataTypeVector& dtypes,
          const std::vector<tensorflow::PartialTensorShape>& shapes)
          : DatasetIterator<Dataset>(params),
            required_keep_alive_(required_keep_alive),
            configs_(configs),
            clear_after_episode_(clear_after_episode) {}

      tensorflow::Status Initialize(
          tensorflow::data::IteratorContext* ctx) override {
        structured_writer_ = std::make_unique<StructuredWriter>(
            std::make_unique<QueueWriter>(required_keep_alive_, &data_),
            configs_);
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, this, prefix(), &input_iter_));
        return dataset()->captured_func_->Instantiate(
            ctx, &instantiated_captured_func_);
      }

      tensorflow::Status GetNextInternal(
          tensorflow::data::IteratorContext* ctx,
          std::vector<tensorflow::Tensor>* out_tensors,
          bool* end_of_sequence) override {
        // This needs to be thread-safe.
        // We lock the full method because otherwise we would have several
        // threads getting data from the input dataset and inserting into the
        // queue.
        absl::MutexLock lock(&mu_);
        while (data_.empty()) {
          std::vector<tensorflow::Tensor> out_steps;
          bool input_end_of_sequence = false;
          TF_RETURN_IF_ERROR(
              input_iter_->GetNext(ctx, &out_steps, &input_end_of_sequence));
          if (input_end_of_sequence) {
            break;
          }

          // Checks if this is the end of the episode
          std::vector<tensorflow::Tensor> result;
          TF_RETURN_IF_ERROR(instantiated_captured_func_->RunWithBorrowedArgs(
              ctx, out_steps, &result, model_node()));

          if (result.size() != 1 || result[0].dtype() != tensorflow::DT_BOOL ||
              result[0].NumElements() != 1) {
            return tensorflow::errors::InvalidArgument(
                "Function `is_end_of_episode` must return a scalar bool.");
          }
          bool end_episode = result[0].scalar<bool>()();
          std::vector<absl::optional<tensorflow::Tensor>> optional_out_steps;
          optional_out_steps.reserve(out_steps.size());
          for (auto step : out_steps) {
            optional_out_steps.push_back(std::move(step));
          }
          TF_RETURN_IF_ERROR(ToTensorflowStatus(
              structured_writer_->Append(optional_out_steps)));
          if (end_episode) {
            TF_RETURN_IF_ERROR(ToTensorflowStatus(
                structured_writer_->EndEpisode(clear_after_episode_)));
          }
        }
        if (data_.empty()) {
          // There is no more data in the input dataset.
          *end_of_sequence = true;
        } else {
          *out_tensors = data_.front();
          data_.pop_front();
        }
        return tensorflow::OkStatus();
      }

     protected:
      tensorflow::Status SaveInternal(
          tensorflow::data::SerializationContext* ctx,
          tensorflow::data::IteratorStateWriter* writer) override {
        return tensorflow::errors::Unimplemented(
            "SaveInternal is currently not supported");
      }

      tensorflow::Status RestoreInternal(
          tensorflow::data::IteratorContext* ctx,
          tensorflow::data::IteratorStateReader* reader) override {
        return tensorflow::errors::Unimplemented(
            "RestoreInternal is currently not supported");
      }

     private:
      const int required_keep_alive_;
      const std::vector<StructuredWriterConfig> configs_;
      const bool clear_after_episode_;
      std::unique_ptr<tensorflow::data::InstantiatedCapturedFunction>
          instantiated_captured_func_;
      std::unique_ptr<StructuredWriter> structured_writer_;
      std::deque<std::vector<tensorflow::Tensor>> data_;
      std::unique_ptr<IteratorBase> input_iter_;
      // Protects GetNextInternal.
      absl::Mutex mu_;
    };  // Iterator.

    const DatasetBase* const input_;
    const int required_keep_alive_;
    const std::vector<StructuredWriterConfig> configs_;
    const bool clear_after_episode_;
    const std::unique_ptr<tensorflow::data::CapturedFunction> captured_func_;
    const tensorflow::DataTypeVector dtypes_;
    const std::vector<tensorflow::PartialTensorShape> shapes_;
  };  // Dataset.

  int required_keep_alive_;
  std::vector<StructuredWriterConfig> configs_;
  tensorflow::DataTypeVector dtypes_;
  std::vector<tensorflow::PartialTensorShape> shapes_;
  bool clear_after_episode_;
  std::shared_ptr<tensorflow::data::FunctionMetadata> func_metadata_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(ReverbPatternDatasetOp);
};

REGISTER_KERNEL_BUILDER(
    Name("ReverbPatternDataset").Device(tensorflow::DEVICE_CPU),
    ReverbPatternDatasetOp);

}  // namespace
}  // namespace reverb
}  // namespace deepmind
