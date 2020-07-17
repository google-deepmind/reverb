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

#include "reverb/cc/sampler.h"

#include <algorithm>
#include <string>

#include "grpcpp/impl/codegen/client_context.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include "absl/time/time.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/tensor_compression.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace deepmind {
namespace reverb {
namespace {

inline bool SampleIsDone(const std::vector<SampleStreamResponse>& sample) {
  if (sample.empty()) return false;
  int64_t chunk_length = 0;
  for (const auto& response : sample) {
    chunk_length += response.data().data(0).tensor_shape().dim(0).size();
  }
  const auto& range = sample.front().info().item().sequence_range();
  return chunk_length >= range.length() + range.offset();
}

template <typename T>
tensorflow::Tensor InitializeTensor(T value, int64_t length) {
  tensorflow::Tensor tensor(tensorflow::DataTypeToEnum<T>::v(),
                            tensorflow::TensorShape({length}));
  auto tensor_t = tensor.flat<T>();
  std::fill(tensor_t.data(), tensor_t.data() + length, value);
  return tensor;
}

tensorflow::Status AsSample(std::vector<SampleStreamResponse> responses,
                            std::unique_ptr<Sample>* sample) {
  const auto& info = responses.front().info();

  // Extract all chunks belonging to this sample.
  std::list<std::vector<tensorflow::Tensor>> chunks;

  // The chunks are not required to be aligned perfectly with the data so a
  // part of the first chunk is potentially stripped. The same applies to the
  // last part of the final chunk.
  int64_t offset = info.item().sequence_range().offset();
  int64_t remaining = info.item().sequence_range().length();

  for (auto& response : responses) {
    REVERB_CHECK_GT(remaining, 0);

    std::vector<tensorflow::Tensor> batches;
    batches.resize(response.data().data_size());

    int64_t batch_size = -1;

    // Convert each chunk tensor and release the chunk memory afterwards.
    int64_t insert_index = response.data().data_size() - 1;
    while (!response.data().data().empty()) {
      tensorflow::Tensor batch;

      {
        // This ensures we release the response proto after converting the
        // result to a tensor.
        auto chunk = absl::WrapUnique(
            response.mutable_data()->mutable_data()->ReleaseLast());
        batch = DecompressTensorFromProto(*chunk);
      }

      if (response.data().delta_encoded()) {
        batch = DeltaEncode(batch, /*encode=*/false);
      }

      if (batch_size < 0) {
        batch_size = batch.dim_size(0);
      } else {
        if (batch_size != batch.dim_size(0)) {
          return tensorflow::errors::Internal(
              "Chunks of the same response must have identical batch size, but "
              "first chunk has batch size ",
              batch_size, " while the current chunk has batch size ",
              batch.dim_size(0));
        }
      }

      batch =
          batch.Slice(offset, std::min<int64_t>(offset + remaining, batch_size));
      if (!batch.IsAligned()) {
        batch = tensorflow::tensor::DeepCopy(batch);
      }

      batches[insert_index--] = std::move(batch);
    }

    chunks.push_back(std::move(batches));

    remaining -= std::min<int64_t>(remaining, batch_size - offset);
    offset = 0;
  }

  REVERB_CHECK_EQ(remaining, 0);

  *sample = absl::make_unique<Sample>(info.item().key(), info.probability(),
                                      info.table_size(), info.item().priority(),
                                      std::move(chunks));
  return tensorflow::Status::OK();
}

}  // namespace

Sampler::Sampler(std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
                 const std::string& table, const Options& options,
                 internal::DtypesAndShapes dtypes_and_shapes)
    : stub_(std::move(stub)),
      table_(table),
      max_samples_(options.max_samples == kUnlimitedMaxSamples
                       ? INT64_MAX
                       : options.max_samples),
      max_samples_per_stream_(options.max_samples_per_stream == kAutoSelectValue
                                  ? kDefaultMaxSamplesPerStream
                                  : options.max_samples_per_stream),
      rate_limiter_timeout_(options.rate_limiter_timeout),
      active_sample_(nullptr),
      samples_(std::max<int>(options.num_workers, 1)),
      dtypes_and_shapes_(std::move(dtypes_and_shapes)) {
  REVERB_CHECK_GT(max_samples_, 0);
  REVERB_CHECK_GT(options.max_in_flight_samples_per_worker, 0);
  REVERB_CHECK(options.num_workers == kAutoSelectValue ||
               options.num_workers > 0);
  REVERB_CHECK(options.flexible_batch_size == kAutoSelectValue ||
               options.flexible_batch_size > 0);

  int64_t num_workers = options.num_workers == kAutoSelectValue
                          ? kDefaultNumWorkers
                          : options.num_workers;

  // If a subset of the workers are able to fetch all of `max_samples_` in the
  // first batch then there is no point in creating all of them.
  num_workers = std::min<int64_t>(
      num_workers,
      std::max<int64_t>(1,
                      max_samples_ / options.max_in_flight_samples_per_worker));

  for (int i = 0; i < num_workers; i++) {
    workers_.push_back(absl::make_unique<Worker>(
        stub_, table, options.max_in_flight_samples_per_worker,
        options.flexible_batch_size));
    worker_threads_.push_back(internal::StartThread(
        absl::StrCat("SampleWorker", i),
        [this, worker = workers_[i].get()] { RunWorker(worker); }));
  }
}

Sampler::~Sampler() { Close(); }

tensorflow::Status Sampler::GetNextTimestep(
    std::vector<tensorflow::Tensor>* data, bool* end_of_sequence) {
  TF_RETURN_IF_ERROR(MaybeSampleNext());

  *data = active_sample_->GetNextTimestep();
  TF_RETURN_IF_ERROR(ValidateAgainstOutputSpec(*data, /*time_step=*/true));

  if (end_of_sequence != nullptr) {
    *end_of_sequence = active_sample_->is_end_of_sample();
  }

  if (active_sample_->is_end_of_sample()) {
    absl::WriterMutexLock lock(&mu_);
    if (++returned_ == max_samples_) samples_.Close();
  }

  return tensorflow::Status::OK();
}

tensorflow::Status Sampler::GetNextSample(
    std::vector<tensorflow::Tensor>* data) {
  std::unique_ptr<Sample> sample;
  TF_RETURN_IF_ERROR(PopNextSample(&sample));
  TF_RETURN_IF_ERROR(sample->AsBatchedTimesteps(data));
  TF_RETURN_IF_ERROR(ValidateAgainstOutputSpec(*data, /*time_step=*/false));

  absl::WriterMutexLock lock(&mu_);
  if (++returned_ == max_samples_) samples_.Close();
  return tensorflow::Status::OK();
}

tensorflow::Status Sampler::ValidateAgainstOutputSpec(
    const std::vector<tensorflow::Tensor>& data, bool time_step) {
  if (!dtypes_and_shapes_) {
    return tensorflow::Status::OK();
  }

  if (data.size() != dtypes_and_shapes_->size()) {
    return tensorflow::errors::InvalidArgument(
        "Inconsistent number of tensors received from table '", table_,
        "'.  Specification has ", dtypes_and_shapes_->size(),
        " tensors, but data coming from the table shows ", data.size(),
        " tensors.\nTable signature: ",
        internal::DtypesShapesString(*dtypes_and_shapes_),
        ".\nIncoming tensor signature: ",
        internal::DtypesShapesString(internal::SpecsFromTensors(data)));
  }

  for (int i = 0; i < data.size(); ++i) {
    tensorflow::TensorShape elem_shape;
    if (!time_step) {
      // Remove the outer dimension from data[i].shape() so we can properly
      // compare against the spec (which doesn't have the sequence dimension).
      elem_shape = data[i].shape();
      if (elem_shape.dims() == 0) {
        return tensorflow::errors::InvalidArgument(
            "Invalid tensor shape received from table '", table_,
            "'.  "
            "time_step is false but data[",
            i,
            "] has scalar shape "
            "(no time dimension).");
      }
      elem_shape.RemoveDim(0);
    }

    auto* shape_ptr = time_step ? &(data[i].shape()) : &elem_shape;
    if (data[i].dtype() != dtypes_and_shapes_->at(i).dtype ||
        !dtypes_and_shapes_->at(i).shape.IsCompatibleWith(*shape_ptr)) {
      return tensorflow::errors::InvalidArgument(
          "Received incompatible tensor at flattened index ", i,
          " from table '", table_, "'.  Specification has (dtype, shape): (",
          tensorflow::DataTypeString(dtypes_and_shapes_->at(i).dtype), ", ",
          dtypes_and_shapes_->at(i).shape.DebugString(),
          ").  Tensor has (dtype, shape): (",
          tensorflow::DataTypeString(data[i].dtype()), ", ",
          shape_ptr->DebugString(), ").\nTable signature: ",
          internal::DtypesShapesString(*dtypes_and_shapes_));
    }
  }
  return tensorflow::Status::OK();
}

bool Sampler::should_stop_workers() const {
  return closed_ || returned_ == max_samples_ || !stream_status_.ok();
}

void Sampler::Close() {
  {
    absl::WriterMutexLock lock(&mu_);
    if (closed_) return;
    closed_ = true;
  }

  for (auto& worker : workers_) {
    worker->Cancel();
  }

  samples_.Close();
  worker_threads_.clear();  // Joins worker threads.
}

tensorflow::Status Sampler::MaybeSampleNext() {
  if (active_sample_ != nullptr && !active_sample_->is_end_of_sample()) {
    return tensorflow::Status::OK();
  }

  return PopNextSample(&active_sample_);
}

tensorflow::Status Sampler::PopNextSample(std::unique_ptr<Sample>* sample) {
  if (samples_.Pop(sample)) return tensorflow::Status::OK();

  absl::ReaderMutexLock lock(&mu_);
  if (returned_ == max_samples_) {
    return tensorflow::errors::OutOfRange("`max_samples` already returned.");
  }
  if (closed_) {
    return tensorflow::errors::Cancelled("Sampler has been cancelled.");
  }
  return FromGrpcStatus(stream_status_);
}

void Sampler::RunWorker(Worker* worker) {
  auto trigger = [this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return should_stop_workers() || requested_ < max_samples_;
  };

  while (true) {
    mu_.LockWhen(absl::Condition(&trigger));

    if (should_stop_workers()) {
      mu_.Unlock();
      return;
    }
    int64_t samples_to_stream =
        std::min<int64_t>(max_samples_per_stream_, max_samples_ - requested_);
    requested_ += samples_to_stream;
    mu_.Unlock();

    auto result = worker->OpenStreamAndFetch(&samples_, samples_to_stream,
                                             rate_limiter_timeout_);

    {
      absl::WriterMutexLock lock(&mu_);

      // If the stream was closed prematurely then we need to reduce the number
      // of requested samples by the difference of the expected number and the
      // actual.
      requested_ -= samples_to_stream - result.first;

      // Overwrite the final status only if it wasn't already an error.
      if (stream_status_.ok() && !result.second.ok() &&
          result.second.error_code() != grpc::StatusCode::UNAVAILABLE) {
        stream_status_ = result.second;
        samples_.Close();  // Unblock any pending calls.
        return;
      }
    }
  }
}

Sampler::Worker::Worker(
    std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
    std::string table, int64_t samples_per_request, int32_t flexible_batch_size)
    : stub_(std::move(stub)),
      table_(std::move(table)),
      samples_per_request_(samples_per_request),
      flexible_batch_size_(flexible_batch_size) {}

std::pair<int64_t, grpc::Status> Sampler::Worker::OpenStreamAndFetch(
    deepmind::reverb::internal::Queue<std::unique_ptr<Sample>>* queue,
    int64_t num_samples, absl::Duration rate_limiter_timeout) {
  std::unique_ptr<grpc::ClientReaderWriterInterface<SampleStreamRequest,
                                                    SampleStreamResponse>>
      stream;
  {
    absl::MutexLock lock(&mu_);
    if (closed_) {
      return {0, grpc::Status(grpc::StatusCode::CANCELLED,
                              "`Close` called on Sampler.")};
    }
    context_ = absl::make_unique<grpc::ClientContext>();
    context_->set_wait_for_ready(false);
    stream = stub_->SampleStream(context_.get());
  }

  int64_t num_samples_returned = 0;
  while (num_samples_returned < num_samples) {
    SampleStreamRequest request;
    request.set_table(table_);
    request.set_num_samples(
        std::min(samples_per_request_, num_samples - num_samples_returned));
    request.mutable_rate_limiter_timeout()->set_milliseconds(
        NonnegativeDurationToInt64Millis(rate_limiter_timeout));
    request.set_flexible_batch_size(flexible_batch_size_);

    if (!stream->Write(request)) {
      return {num_samples_returned, stream->Finish()};
    }

    for (int64_t i = 0; i < request.num_samples(); i++) {
      std::vector<SampleStreamResponse> responses;
      while (!SampleIsDone(responses)) {
        SampleStreamResponse response;
        if (!stream->Read(&response)) {
          return {num_samples_returned, stream->Finish()};
        }
        responses.push_back(std::move(response));
      }

      std::unique_ptr<Sample> sample;
      auto status = AsSample(std::move(responses), &sample);
      if (!status.ok()) {
        return {num_samples_returned, ToGrpcStatus(status)};
      }
      if (!queue->Push(std::move(sample))) {
        return {num_samples_returned,
                grpc::Status(grpc::StatusCode::CANCELLED,
                             "`Close` called on Sampler.")};
      }
      ++num_samples_returned;
    }
  }

  if (num_samples_returned != num_samples) {
    return {
        num_samples_returned,
        grpc::Status(grpc::StatusCode::INTERNAL,
                     absl::StrCat("num_samples_returned != num_samples (",
                                  num_samples_returned, " vs. ", num_samples))};
  }
  return {num_samples_returned, grpc::Status::OK};
}

void Sampler::Worker::Cancel() {
  absl::MutexLock lock(&mu_);
  closed_ = true;
  if (context_ != nullptr) context_->TryCancel();
}

Sample::Sample(tensorflow::uint64 key, double probability,
               tensorflow::int64 table_size, double priority,
               std::list<std::vector<tensorflow::Tensor>> chunks)
    : key_(key),
      probability_(probability),
      table_size_(table_size),
      priority_(priority),
      num_timesteps_(0),
      num_data_tensors_(0),
      chunks_(std::move(chunks)),
      next_timestep_index_(0),
      next_timestep_called_(false) {
  REVERB_CHECK(!chunks_.empty()) << "Must provide at least one chunk.";
  REVERB_CHECK(!chunks_.front().empty())
      << "Chunks must hold at least one tensor.";

  num_data_tensors_ = chunks_.front().size();
  for (const auto& batches : chunks_) {
    num_timesteps_ += batches.front().dim_size(0);
  }
}

std::vector<tensorflow::Tensor> Sample::GetNextTimestep() {
  REVERB_CHECK(!is_end_of_sample());

  // Construct the output tensors.
  std::vector<tensorflow::Tensor> result;
  result.reserve(num_data_tensors_ + 4);
  result.push_back(tensorflow::Tensor(key_));
  result.push_back(tensorflow::Tensor(probability_));
  result.push_back(tensorflow::Tensor(table_size_));
  result.push_back(tensorflow::Tensor(priority_));

  for (const auto& t : chunks_.front()) {
    auto slice = t.SubSlice(next_timestep_index_);
    if (slice.IsAligned()) {
      result.push_back(std::move(slice));
    } else {
      result.push_back(tensorflow::tensor::DeepCopy(slice));
    }
  }

  // Advance the iterator.
  ++next_timestep_index_;
  if (next_timestep_index_ == chunks_.front().front().dim_size(0)) {
    // Go to the next chunk.
    chunks_.pop_front();
    next_timestep_index_ = 0;
  }
  next_timestep_called_ = true;

  return result;
}

bool Sample::is_end_of_sample() const { return chunks_.empty(); }

tensorflow::Status Sample::AsBatchedTimesteps(
    std::vector<tensorflow::Tensor>* data) {
  if (next_timestep_called_) {
    return tensorflow::errors::DataLoss(
        "Sample::AsBatchedTimesteps: Some time steps have been lost.");
  }

  std::vector<tensorflow::Tensor> sequences(num_data_tensors_ + 4);

  // Initialize the first three items with the key, probability and table size.
  sequences[0] = InitializeTensor(key_, num_timesteps_);
  sequences[1] = InitializeTensor(probability_, num_timesteps_);
  sequences[2] = InitializeTensor(table_size_, num_timesteps_);
  sequences[3] = InitializeTensor(priority_, num_timesteps_);

  // Prepare the data for concatenation.
  // data_tensors[i][j] is the j-th chunk of the i-th data tensor.
  std::vector<std::vector<tensorflow::Tensor>> data_tensors(num_data_tensors_);

  // Extract all chunks.
  while (!chunks_.empty()) {
    auto it_to = data_tensors.begin();
    for (auto& batch : chunks_.front()) {
      (it_to++)->push_back(std::move(batch));
    }
    chunks_.pop_front();
  }

  // Concatenate all chunks.
  int64_t i = 4;
  for (const auto& chunks : data_tensors) {
    TF_RETURN_IF_ERROR(tensorflow::tensor::Concat(chunks, &sequences[i++]));
  }

  std::swap(sequences, *data);

  return tensorflow::Status::OK();
}

tensorflow::Status Sampler::Options::Validate() const {
  if (max_samples < 1 && max_samples != kUnlimitedMaxSamples) {
    return tensorflow::errors::InvalidArgument(
        "max_samples (", max_samples, ") must be ", kUnlimitedMaxSamples,
        " or >= 1");
  }
  if (max_in_flight_samples_per_worker < 1) {
    return tensorflow::errors::InvalidArgument(
        "max_in_flight_samples_per_worker (", max_in_flight_samples_per_worker,
        ") has to be >= 1");
  }
  if (num_workers < 1 && num_workers != kAutoSelectValue) {
    return tensorflow::errors::InvalidArgument("num_workers (", num_workers,
                                               ") must be ", kAutoSelectValue,
                                               " or >= 1");
  }
  if (max_samples_per_stream < 1 &&
      max_samples_per_stream != kUnlimitedMaxSamples) {
    return tensorflow::errors::InvalidArgument(
        "max_samples_per_stream (", max_samples_per_stream, ") must be ",
        kAutoSelectValue, " or >= 1");
  }
  if (rate_limiter_timeout < absl::ZeroDuration()) {
    return tensorflow::errors::InvalidArgument("rate_limiter_timeout (",
                                               rate_limiter_timeout,
                                               ") must not be negative.");
  }
  return tensorflow::Status::OK();
}

}  // namespace reverb
}  // namespace deepmind
