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
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "grpcpp/impl/codegen/client_context.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/errors.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/tf_util.h"
#include "reverb/cc/support/trajectory_util.h"
#include "reverb/cc/table.h"
#include "reverb/cc/tensor_compression.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"

namespace deepmind {
namespace reverb {
namespace {

// Initial sampling batch size is kept relatively low to not allocate a lot
// of memory for the response when table is close to empty and only a few items
// can be fetched.
constexpr int64_t kInitialSampleBatchSize = 8;

template <typename T>
tensorflow::Tensor InitializeTensor(T value, int64_t length) {
  tensorflow::Tensor tensor(tensorflow::DataTypeToEnum<T>::v(),
                            tensorflow::TensorShape({length}));
  auto tensor_t = tensor.flat<T>();
  std::fill(tensor_t.data(), tensor_t.data() + length, value);
  return tensor;
}

template <typename T>
tensorflow::Tensor ScalarTensor(T value) {
  // TODO(b/186669968): Move to the Tensor(scalar_value) constructor once
  // alignment bug is fixed.
  tensorflow::Tensor tensor(tensorflow::DataTypeToEnum<T>::v(),
                            tensorflow::TensorShape({}));
  tensor.scalar<T>()() = std::move(value);
  return tensor;
}

absl::Status AsSample(std::vector<SampleStreamResponse::SampleEntry> responses,
                      std::unique_ptr<Sample>* sample) {
  const auto& info = responses.front().info();
  internal::flat_hash_map<uint64_t, std::unique_ptr<ChunkData>> chunks;
  for (auto& response : responses) {
    while (response.data_size() != 0) {
      auto* chunk = response.mutable_data()->ReleaseLast();
      chunks[chunk->chunk_key()] = absl::WrapUnique<ChunkData>(chunk);
    }
  }

  // Count the number of times each chunk is referenced in the column slices.
  // This allows us to check if the chunk is needed anymore after every use. If
  // all the references have been handled then the memory of the chunk can be
  // freed thus reducing total memory usage.
  internal::flat_hash_map<uint64_t, int> chunk_ref_count;
  for (const auto& column : info.item().flat_trajectory().columns()) {
    for (const auto& slice : column.chunk_slices()) {
      chunk_ref_count[slice.chunk_key()]++;
    }
  }

  // Extract all chunks belonging to this sample.
  const auto& columns = info.item().flat_trajectory().columns();

  std::vector<std::vector<tensorflow::Tensor>> column_chunks(columns.size());
  std::vector<bool> squeeze_columns(columns.size());

  for (int i = 0; i < columns.size(); i++) {
    squeeze_columns[i] = columns[i].squeeze();
    for (const auto& slice : columns[i].chunk_slices()) {
      auto it = chunks.find(slice.chunk_key());
      if (it == chunks.end()) {
        return absl::InternalError(
            absl::StrCat("Chunk ", slice.chunk_key(),
                         " could not be found when unpacking item ",
                         info.item().key(), "."));
      }

      column_chunks[i].emplace_back();
      REVERB_RETURN_IF_ERROR(internal::UnpackChunkColumnAndSlice(
          *it->second, slice, &column_chunks[i].back()));

      // If this was the last time the chunk is referenced the we can release
      // its memory.
      if (--chunk_ref_count[slice.chunk_key()] == 0) {
        chunks.erase(it);
      }
    }
  }

  *sample = absl::make_unique<Sample>(
      info.item().key(), info.probability(), info.table_size(),
      info.item().priority(), info.rate_limited(), std::move(column_chunks),
      std::move(squeeze_columns));

  return absl::OkStatus();
}

absl::Status AsSample(const Table::SampledItem& sampled_item,
                      std::unique_ptr<Sample>* sample) {
  internal::flat_hash_map<uint64_t, std::shared_ptr<ChunkStore::Chunk>> chunks(
      sampled_item.ref->chunks.size());
  for (auto& chunk : sampled_item.ref->chunks) {
    chunks[chunk->key()] = chunk;
  }

  std::vector<std::vector<tensorflow::Tensor>> column_chunks;
  column_chunks.reserve(
      sampled_item.ref->item.flat_trajectory().columns_size());

  for (const auto& column :
       sampled_item.ref->item.flat_trajectory().columns()) {
    std::vector<tensorflow::Tensor> unpacked_chunks;

    for (const auto& slice : column.chunk_slices()) {
      unpacked_chunks.emplace_back();
      REVERB_RETURN_IF_ERROR(internal::UnpackChunkColumnAndSlice(
          chunks[slice.chunk_key()]->data(), slice, &unpacked_chunks.back()));
    }

    column_chunks.push_back(std::move(unpacked_chunks));
  }

  std::vector<bool> squeeze_columns;
  for (const auto& col : sampled_item.ref->item.flat_trajectory().columns()) {
    squeeze_columns.push_back(col.squeeze());
  }
  *sample = absl::make_unique<deepmind::reverb::Sample>(
      sampled_item.ref->item.key(), sampled_item.probability,
      sampled_item.table_size, sampled_item.priority, sampled_item.rate_limited,
      std::move(column_chunks), std::move(squeeze_columns));

  return absl::OkStatus();
}

class GrpcSamplerWorker : public SamplerWorker {
 public:
  // Constructs a new worker without creating a stream to a server.
  GrpcSamplerWorker(
      std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
      std::string table_name, int64_t samples_per_request)
      : stub_(std::move(stub)),
        table_name_(std::move(table_name)),
        samples_per_request_(samples_per_request),
        reserved_slots_(0) {}

  // Cancels the stream and marks the worker as closed. Active and future
  // calls to `OpenStreamAndFetch` will return status `CANCELLED`.
  void Cancel() override {
    absl::MutexLock lock(&mu_);
    closed_ = true;
    if (context_ != nullptr) context_->TryCancel();
  }

  // Opens a new `SampleStream` to a server and requests `num_samples` samples
  // in batches with maximum size `samples_per_request`, with a timeout to
  // pass to the `Table::Sample` call. Once complete (either
  // done, from a non transient error, or from timing out), the stream is
  // closed and the number of samples pushed to `queue` is returned together
  // with the status of the stream.  A timeout will cause the Status type
  // DeadlineExceeded to be returned.
  std::pair<int64_t, absl::Status> FetchSamples(
      internal::Queue<std::unique_ptr<Sample>>* queue, int64_t num_samples,
      absl::Duration rate_limiter_timeout) override {
    std::unique_ptr<grpc::ClientReaderWriterInterface<SampleStreamRequest,
                                                      SampleStreamResponse>>
        stream;
    {
      absl::MutexLock lock(&mu_);
      if (closed_) {
        return {0, absl::CancelledError("`Close` called on Sampler.")};
      }
      context_ = absl::make_unique<grpc::ClientContext>();
      context_->set_wait_for_ready(false);
      stream = stub_->SampleStream(context_.get());
    }

    int64_t num_samples_returned = 0;
    SampleStreamResponse response;
    while (num_samples_returned < num_samples) {
      // TODO(b/190237214): Ignore timeouts when data is not being requested.
      SampleStreamRequest request;
      request.set_table(table_name_);
      request.set_num_samples(
          std::min(samples_per_request_, num_samples - num_samples_returned));
      request.mutable_rate_limiter_timeout()->set_milliseconds(
          NonnegativeDurationToInt64Millis(rate_limiter_timeout));
      // Reservation can be negative if previously reserved slots are being
      // returned.
      if (!queue->Reserve(request.num_samples() - reserved_slots_)) {
          return {num_samples_returned,
                  absl::CancelledError("`Close` called on Sampler")};
      }
      reserved_slots_ = request.num_samples();

      if (!stream->Write(request)) {
        return {num_samples_returned, FromGrpcStatus(stream->Finish())};
      }

      std::vector<SampleStreamResponse::SampleEntry> parts_of_next_sample;
      for (int64_t sampled = 0; sampled < request.num_samples();) {
        if (!stream->Read(&response)) {
          auto status = FromGrpcStatus(stream->Finish());
          if (errors::IsRateLimiterTimeout(status) &&
              queue->num_waiting_to_pop() < 1) {
            // The rate limiter timed out but no one is waiting for new data,
            // so we can exit with an OkStatus and get restarted with a new
            // stream.
            return {num_samples_returned, absl::OkStatus()};
          } else {
            return {num_samples_returned, status};
          }
        }
        for (auto& entry : response.entries()) {
          parts_of_next_sample.push_back(std::move(entry));
          // Continue grabbing entries until the current sample is complete.
          if (!parts_of_next_sample.back().end_of_sequence()) {
            continue;
          }

          // We have received everything we need to unpack the next sample so
          // let's push it to the queue. We don't expect AsSample to ever fail
          // but it will be closed if the Sampler has been closed.
          std::unique_ptr<Sample> sample;
          auto status = AsSample(std::move(parts_of_next_sample), &sample);
          parts_of_next_sample.clear();
          if (!status.ok()) {
            return {num_samples_returned, status};
          }
          if (--reserved_slots_ < 0) {
            return {num_samples_returned,
                    absl::InternalError(
                        "This should never happen. Please file a bug to the "
                        "Reverb team if you encounter this error.")};
          }
          queue->Push(std::move(sample));
          // The sample was successfully received from the stream and pushed to
          // the queue. There might still be more samples, or partial samples,
          // in the same SampleStreamResponse so we'll continue reading the
          // remaining entries into the next sample.
          ++num_samples_returned;
          ++sampled;
        }
      }
      if (!parts_of_next_sample.empty()) {
        return {num_samples_returned,
                absl::InternalError(
                    "Streamed responses included unattributed SampleEntry.")};
      }
    }

    if (num_samples_returned != num_samples) {
      return {num_samples_returned,
              absl::InternalError(
                  absl::StrCat("num_samples_returned != num_samples (",
                               num_samples_returned, " vs. ", num_samples))};
    }
    return {num_samples_returned, absl::OkStatus()};
  }

 private:
  // Stub used to open `SampleStream`-streams to a server.
  std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub_;

  // Name of the `Table` to sample from.
  const std::string table_name_;

  // The maximum number of samples to request in a "batch".
  const int64_t samples_per_request_;

  // Number of reserved slots in the queue;
  int64_t reserved_slots_;

  // Context of the active stream.
  std::unique_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mu_);

  // True if `Cancel` has been called.
  bool closed_ ABSL_GUARDED_BY(mu_) = false;

  absl::Mutex mu_;
};

class LocalSamplerWorker : public SamplerWorker {
 public:
  // Constructs a new worker without creating a stream to a server.
  LocalSamplerWorker(std::shared_ptr<Table> table,
                     int max_in_flight_samples)
      : table_(table),
        max_in_flight_samples_(max_in_flight_samples),
        reserved_slots_(0) {
    REVERB_CHECK_GE(max_in_flight_samples_, 1);
  }

  void Cancel() override {
    absl::MutexLock lock(&mu_);
    closed_ = true;
  }

  std::pair<int64_t, absl::Status> FetchSamples(
      internal::Queue<std::unique_ptr<Sample>>* queue, int64_t num_samples,
      absl::Duration rate_limiter_timeout) override {
    static const auto kWakeupTimeout = absl::Seconds(3);
    auto final_deadline = absl::Now() + rate_limiter_timeout;

    int64_t num_samples_returned = 0;
    int64_t prev_batch_size = kInitialSampleBatchSize;
    while (num_samples_returned < num_samples) {
      {
        absl::MutexLock lock(&mu_);
        if (closed_) {
          return {0, absl::CancelledError("`Close` called on Sampler.")};
        }
      }

      // If the rate limiter deadline is long into the future then we set the
      // deadline `kWakeupTimeout` from now instead. Periodically waking up
      // allows us to check that the Sampler haven't been cancelled while we
      // were waiting.
      auto timeout =
          std::min(final_deadline, absl::Now() + kWakeupTimeout) - absl::Now();

      // Try to double previously returned response batch size while not
      // exceeding the limits.
      auto batch_size =
          std::min<int64_t>(max_in_flight_samples_,
                          std::min<int64_t>(2 * prev_batch_size,
                                          num_samples - num_samples_returned));
      // Reservation can be negative if previously reserved slots are being
      // returned.
      if (!queue->Reserve(batch_size - reserved_slots_)) {
        return {num_samples_returned,
                absl::CancelledError("`Close` called on Sampler")};
      }
      reserved_slots_ = batch_size;
      std::vector<Table::SampledItem> items;
      auto status = table_->SampleFlexibleBatch(&items, batch_size, timeout);

      // If the deadline is exceeded but the "real deadline" is still in the
      // future then we are only waking up to check for cancellation.
      if (absl::IsDeadlineExceeded(status)) {
        if (absl::Now() < final_deadline) {
          continue;
        }
        if (queue->num_waiting_to_pop() < 1) {
          // While no items requested, we reset the final_deadline and restart.
          final_deadline = absl::Now() + rate_limiter_timeout;
          continue;
        }
      }

      // All other errors are "real" and thus should be returned to the caller.
      if (!status.ok()) {
        return {num_samples_returned, status};
      }
      // Update `prev_batch_size` after handling timeouts and errors. Otherwise
      // batch size would get reset to a small value affecting performance.
      prev_batch_size = items.size();

      // We received new items, so reset the timeout deadline.
      final_deadline = absl::Now() + rate_limiter_timeout;

      // Push sampled items to queue.
      for (const auto& item : items) {
        std::unique_ptr<Sample> sample;
        if (status = AsSample(item, &sample); !status.ok()) {
          return {num_samples_returned, status};
        }
        if (--reserved_slots_ < 0) {
          return {num_samples_returned,
                  absl::InternalError(
                      "This should never happen. Please file a bug to the "
                      "Reverb team if you encounter this error.")};
        }
        queue->Push(std::move(sample));
        ++num_samples_returned;
      }
    }

    if (num_samples_returned != num_samples) {
      return {num_samples_returned,
              absl::InternalError(
                  absl::StrCat("num_samples_returned != num_samples (",
                               num_samples_returned, " vs. ", num_samples))};
    }
    return {num_samples_returned, absl::OkStatus()};
  }

 private:
  std::shared_ptr<Table> table_;
  const int64_t max_in_flight_samples_;
  int64_t reserved_slots_;
  bool closed_ ABSL_GUARDED_BY(mu_) = false;
  absl::Mutex mu_;
};

int64_t GetNumWorkers(const Sampler::Options& options) {
  int64_t max_samples = options.max_samples == Sampler::kUnlimitedMaxSamples
                          ? INT64_MAX
                          : options.max_samples;
  int64_t num_workers = options.num_workers == Sampler::kAutoSelectValue
                          ? Sampler::kDefaultNumWorkers
                          : options.num_workers;

  // If a subset of the workers are able to fetch all of `max_samples` in the
  // first batch then there is no point in creating all of them.
  return std::min<int64_t>(
      num_workers,
      std::max<int64_t>(1,
                      max_samples / options.max_in_flight_samples_per_worker));
}

std::vector<std::unique_ptr<SamplerWorker>> MakeGrpcWorkers(
    std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
    const std::string& table_name, const Sampler::Options& options) {
  int64_t num_workers = GetNumWorkers(options);
  REVERB_CHECK_GE(num_workers, 1);
  std::vector<std::unique_ptr<SamplerWorker>> workers;
  workers.reserve(num_workers);
  for (int i = 0; i < num_workers; i++) {
    workers.push_back(absl::make_unique<GrpcSamplerWorker>(
        stub, table_name, options.max_in_flight_samples_per_worker));
  }

  return workers;
}

std::vector<std::unique_ptr<SamplerWorker>> MakeLocalWorkers(
    std::shared_ptr<Table> table, const Sampler::Options& options) {
  int64_t num_workers = GetNumWorkers(options);
  REVERB_CHECK_GE(num_workers, 1);

  std::vector<std::unique_ptr<SamplerWorker>> workers;
  workers.reserve(num_workers);
  for (int i = 0; i < num_workers; ++i) {
    workers.push_back(absl::make_unique<LocalSamplerWorker>(
        table, options.max_in_flight_samples_per_worker));
  }
  return workers;
}

}  // namespace

Sampler::Sampler(std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
                 const std::string& table_name, const Options& options,
                 internal::DtypesAndShapes dtypes_and_shapes)
    : Sampler(MakeGrpcWorkers(std::move(stub), table_name, options), table_name,
              options, std::move(dtypes_and_shapes)) {}

Sampler::Sampler(std::vector<std::unique_ptr<SamplerWorker>> workers,
                 const std::string& table, const Options& options,
                 internal::DtypesAndShapes dtypes_and_shapes)
    : table_(table),
      max_samples_(options.max_samples == kUnlimitedMaxSamples
                       ? INT64_MAX
                       : options.max_samples),
      max_samples_per_stream_(options.max_samples_per_stream == kAutoSelectValue
                                  ? kDefaultMaxSamplesPerStream
                                  : options.max_samples_per_stream),
      rate_limiter_timeout_(options.rate_limiter_timeout),
      workers_(std::move(workers)),
      active_sample_(nullptr),
      samples_(options.max_in_flight_samples_per_worker *
               GetNumWorkers(options)),
      dtypes_and_shapes_(std::move(dtypes_and_shapes)) {
  REVERB_CHECK_GT(max_samples_, 0);
  REVERB_CHECK_GT(options.max_in_flight_samples_per_worker, 0);
  REVERB_CHECK(options.num_workers == kAutoSelectValue ||
               options.num_workers > 0);

  for (int i = 0; i < workers_.size(); i++) {
    worker_threads_.push_back(internal::StartThread(
        absl::StrCat("SamplerWorker_", i),
        [this, worker = workers_[i].get()] { RunWorker(worker); }));
  }
}

Sampler::Sampler(std::shared_ptr<Table> table, const Options& options,
                 internal::DtypesAndShapes dtypes_and_shapes)
    : Sampler(MakeLocalWorkers(table, options), table->name(), options,
              std::move(dtypes_and_shapes)) {}

Sampler::~Sampler() { Close(); }

absl::Status Sampler::GetNextTimestep(std::vector<tensorflow::Tensor>* data,
                                      bool* end_of_sequence,
                                      bool* rate_limited) {
  REVERB_RETURN_IF_ERROR(MaybeSampleNext());
  if (!active_sample_->is_composed_of_timesteps()) {
    return absl::InvalidArgumentError(
        "Sampled trajectory cannot be decomposed into timesteps.");
  }

  if (rate_limited != nullptr) {
    *rate_limited = active_sample_->rate_limited();
  }

  *data = active_sample_->GetNextTimestep();
  REVERB_RETURN_IF_ERROR(
      ValidateAgainstOutputSpec(*data, ValidationMode::kTimestep));

  if (end_of_sequence != nullptr) {
    *end_of_sequence = active_sample_->is_end_of_sample();
  }

  if (active_sample_->is_end_of_sample()) {
    absl::WriterMutexLock lock(&mu_);
    if (++returned_ == max_samples_) samples_.Close();
  }

  return absl::OkStatus();
}

absl::Status Sampler::GetNextSample(std::vector<tensorflow::Tensor>* data,
                                    bool* rate_limited) {
  std::unique_ptr<Sample> sample;
  REVERB_RETURN_IF_ERROR(PopNextSample(&sample));
  REVERB_RETURN_IF_ERROR(sample->AsBatchedTimesteps(data));
  REVERB_RETURN_IF_ERROR(
      ValidateAgainstOutputSpec(*data, ValidationMode::kBatchedTimestep));

  if (rate_limited != nullptr) {
    *rate_limited = sample->rate_limited();
  }

  absl::WriterMutexLock lock(&mu_);
  if (++returned_ == max_samples_) samples_.Close();
  return absl::OkStatus();
}

absl::Status Sampler::GetNextTrajectory(std::vector<tensorflow::Tensor>* data,
                                        bool* rate_limited) {
  std::unique_ptr<Sample> sample;
  REVERB_RETURN_IF_ERROR(PopNextSample(&sample));
  REVERB_RETURN_IF_ERROR(sample->AsTrajectory(data));
  REVERB_RETURN_IF_ERROR(
      ValidateAgainstOutputSpec(*data, ValidationMode::kTrajectory));

  if (rate_limited != nullptr) {
    *rate_limited = sample->rate_limited();
  }

  absl::WriterMutexLock lock(&mu_);
  if (++returned_ == max_samples_) samples_.Close();
  return absl::OkStatus();
}

absl::Status Sampler::ValidateAgainstOutputSpec(
    const std::vector<tensorflow::Tensor>& data, Sampler::ValidationMode mode) {
  if (!dtypes_and_shapes_) {
    return absl::OkStatus();
  }

  if (data.size() != dtypes_and_shapes_->size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Inconsistent number of tensors received from table '", table_,
        "'.  Specification has ", dtypes_and_shapes_->size(),
        " tensors, but data coming from the table shows ", data.size(),
        " tensors.\nTable signature: ",
        internal::DtypesShapesString(*dtypes_and_shapes_),
        ".\nIncoming tensor signature: ",
        internal::DtypesShapesString(internal::SpecsFromTensors(data))));
  }

  for (int i = 4; i < data.size(); ++i) {
    tensorflow::TensorShape elem_shape;
    if (mode == ValidationMode::kBatchedTimestep) {
      // Remove the outer dimension from data[i].shape() so we can properly
      // compare against the spec (which doesn't have the sequence dimension).
      elem_shape = data[i].shape();
      if (elem_shape.dims() == 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid tensor shape received from table '", table_,
                         "'.  "
                         "time_step is false but data[",
                         i,
                         "] has scalar shape "
                         "(no time dimension)."));
      }
      elem_shape.RemoveDim(0);
    }

    auto* shape_ptr =
        mode == ValidationMode::kTimestep || mode == ValidationMode::kTrajectory
            ? &(data[i].shape())
            : &elem_shape;
    if (data[i].dtype() != dtypes_and_shapes_->at(i).dtype ||
        !dtypes_and_shapes_->at(i).shape.IsCompatibleWith(*shape_ptr)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Received incompatible tensor at flattened index ", i,
          " from table '", table_, "'.  Specification has (dtype, shape): (",
          tensorflow::DataTypeString(dtypes_and_shapes_->at(i).dtype), ", ",
          dtypes_and_shapes_->at(i).shape.DebugString(),
          ").  Tensor has (dtype, shape): (",
          tensorflow::DataTypeString(data[i].dtype()), ", ",
          shape_ptr->DebugString(), ").\nTable signature: ",
          internal::DtypesShapesString(*dtypes_and_shapes_)));
    }
  }
  return absl::OkStatus();
}

bool Sampler::should_stop_workers() const {
  return closed_ || returned_ == max_samples_ || !worker_status_.ok();
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

absl::Status Sampler::MaybeSampleNext() {
  if (active_sample_ != nullptr && !active_sample_->is_end_of_sample()) {
    return absl::OkStatus();
  }

  return PopNextSample(&active_sample_);
}

absl::Status Sampler::PopNextSample(std::unique_ptr<Sample>* sample) {
  if (samples_.Pop(sample)) return absl::OkStatus();

  absl::ReaderMutexLock lock(&mu_);
  if (returned_ == max_samples_) {
    return absl::OutOfRangeError("`max_samples` already returned.");
  }
  if (closed_) {
    return absl::CancelledError("Sampler has been cancelled.");
  }
  return worker_status_;
}

void Sampler::RunWorker(SamplerWorker* worker) {
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

    auto result = worker->FetchSamples(&samples_, samples_to_stream,
                                       rate_limiter_timeout_);
    {
      absl::WriterMutexLock lock(&mu_);

      // If the stream was closed prematurely then we need to reduce the number
      // of requested samples by the difference of the expected number and the
      // actual.
      requested_ -= samples_to_stream - result.first;

      // Overwrite the final status only if it wasn't already an error.
      if (worker_status_.ok() && !result.second.ok() &&
          !absl::IsUnavailable(result.second)) {
        worker_status_ = result.second;
        samples_.Close();  // Unblock any pending calls.
        return;
      }
    }
  }
}

Sample::Sample(tensorflow::uint64 key, double probability,
               tensorflow::int64 table_size, double priority, bool rate_limited,
               std::vector<std::vector<tensorflow::Tensor>> column_chunks,
               std::vector<bool> squeeze_columns)
    : key_(key),
      probability_(probability),
      table_size_(table_size),
      priority_(priority),
      rate_limited_(rate_limited),
      num_timesteps_(-1),
      squeeze_columns_(std::move(squeeze_columns)),
      next_timestep_called_(false) {
  REVERB_CHECK(!column_chunks.empty()) << "Must provide at least one chunk.";
  REVERB_CHECK(!column_chunks.front().empty())
      << "Chunks must hold at least one tensor.";

  columns_.reserve(column_chunks.size());
  for (auto& chunks : column_chunks) {
    std::deque<ColumnChunk> slices;
    for (auto& chunk : chunks) {
      slices.push_back({std::move(chunk), 0});
    }
    columns_.push_back(std::move(slices));
  }

  if (is_composed_of_timesteps()) {
    num_timesteps_ = 0;
    for (const auto& column_slice : columns_.front()) {
      // Note that we can safely assume that the tensor is not a scalar since a
      // batch dimension is always added when building a chunk. A scalar would
      // thus be represented as a tensor of shape [1].
      num_timesteps_ += column_slice.tensor.dim_size(0);
    }
  }
}

std::vector<tensorflow::Tensor> Sample::GetNextTimestep() {
  REVERB_CHECK(!is_end_of_sample());
  REVERB_CHECK(is_composed_of_timesteps());

  next_timestep_called_ = true;

  // Construct the output tensors.
  std::vector<tensorflow::Tensor> result;
  result.reserve(columns_.size() + 4);
  result.push_back(ScalarTensor(key_));
  result.push_back(ScalarTensor(probability_));
  result.push_back(ScalarTensor(table_size_));
  result.push_back(ScalarTensor(priority_));

  for (auto& col : columns_) {
    auto slice = col.front().tensor.SubSlice(col.front().offset++);
    if (!slice.IsAligned()) {
      slice = tensorflow::tensor::DeepCopy(slice);
    }
    result.push_back(std::move(slice));

    if (col.front().offset == col.front().tensor.dim_size(0)) {
      col.pop_front();
    }
  }

  return result;
}

bool Sample::is_end_of_sample() const {
  return std::all_of(columns_.begin(), columns_.end(),
                     [](const auto& c) { return c.empty(); });
}

bool Sample::is_composed_of_timesteps() const {
  int prev_column_length = -1;
  for (const auto& col : columns_) {
    int column_length = 0;
    for (const auto& column_slice : col) {
      // Note that we can safely assume that the tensor is not a scalar since a
      // batch dimension is always added when building a chunk. A scalar would
      // thus be represented as a tensor of shape [1].
      column_length += column_slice.tensor.dim_size(0);
    }

    if (prev_column_length != -1 && prev_column_length != column_length) {
      return false;
    }
    prev_column_length = column_length;
  }
  return true;
}

bool Sample::rate_limited() const { return rate_limited_; }

absl::Status Sample::AsBatchedTimesteps(std::vector<tensorflow::Tensor>* data) {
  if (next_timestep_called_) {
    return absl::DataLossError(
        "Sample::AsBatchedTimesteps: Some time steps have been lost.");
  }
  if (!is_composed_of_timesteps()) {
    return absl::FailedPreconditionError(
        "Sample::AsBatchedTimesteps when trajectory cannot be decomposed into "
        "timesteps.");
  }

  std::vector<tensorflow::Tensor> sequences(columns_.size() + 4);

  // Initialize the first three items with the key, probability and table size.
  sequences[0] = InitializeTensor(key_, num_timesteps_);
  sequences[1] = InitializeTensor(probability_, num_timesteps_);
  sequences[2] = InitializeTensor(table_size_, num_timesteps_);
  sequences[3] = InitializeTensor(priority_, num_timesteps_);

  // Unpack the data columns.
  REVERB_RETURN_IF_ERROR(UnpackColumns(&sequences));

  std::swap(sequences, *data);

  return absl::OkStatus();
}

absl::Status Sample::AsTrajectory(std::vector<tensorflow::Tensor>* data) {
  if (next_timestep_called_) {
    return absl::DataLossError(
        "Sample::AsBatchedTimesteps: Some time steps have been lost.");
  }
  std::vector<tensorflow::Tensor> sequences(columns_.size() + 4);

  // Initialize the first four items with the key, probability, table size and
  // priority.
  sequences[0] = ScalarTensor(key_);
  sequences[1] = ScalarTensor(probability_);
  sequences[2] = ScalarTensor(table_size_);
  sequences[3] = ScalarTensor(priority_);

  // Unpack the data columns.
  REVERB_RETURN_IF_ERROR(UnpackColumns(&sequences));

  // Remove batch dimension from squeezed columns.
  for (int i = 0; i < squeeze_columns_.size(); i++) {
    if (!squeeze_columns_[i]) continue;
    if (int batch_dim = sequences[i + 4].shape().dim_size(0); batch_dim != 1) {
      return absl::InternalError(absl::StrCat(
          "Tried to squeeze column with batch size ", batch_dim, "."));
    }

    sequences[i + 4] = sequences[i + 4].SubSlice(0);
    if (!sequences[i + 4].IsAligned()) {
      sequences[i + 4] = tensorflow::tensor::DeepCopy(sequences[i + 4]);
    }
  }

  std::swap(sequences, *data);

  return absl::OkStatus();
}

absl::Status Sample::UnpackColumns(std::vector<tensorflow::Tensor>* data) {
  REVERB_CHECK_EQ(data->size(), columns_.size() + 4);

  int64_t i = 4;
  for (const auto& column : columns_) {
    // If the column is made up of a single batched tensor then there will be no
    // need for concatenation so we can save ourselves a copy by simply moving
    // the one (unpacked) chunk into sequences.
    if (column.size() == 1) {
      data->at(i++) = std::move(column.front().tensor);
    } else {
      std::vector<tensorflow::Tensor> column_tensors;
      column_tensors.reserve(column.size());
      for (auto& slice : column) {
        column_tensors.push_back(std::move(slice.tensor));
      }

      REVERB_RETURN_IF_ERROR(FromTensorflowStatus(
          tensorflow::tensor::Concat(column_tensors, &data->at(i++))));
    }
  }
  return absl::OkStatus();
}

absl::Status Sampler::Options::Validate() const {
  if (max_samples < 1 && max_samples != kUnlimitedMaxSamples) {
    return absl::InvalidArgumentError(
        absl::StrCat("max_samples (", max_samples, ") must be ",
                     kUnlimitedMaxSamples, " or >= 1"));
  }
  if (max_in_flight_samples_per_worker < 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("max_in_flight_samples_per_worker (",
                     max_in_flight_samples_per_worker, ") has to be >= 1"));
  }
  if (num_workers < 1 && num_workers != kAutoSelectValue) {
    return absl::InvalidArgumentError(
        absl::StrCat("num_workers (", num_workers, ") must be ",
                     kAutoSelectValue, " or >= 1"));
  }
  if (max_samples_per_stream < 1 &&
      max_samples_per_stream != kUnlimitedMaxSamples) {
    return absl::InvalidArgumentError(
        absl::StrCat("max_samples_per_stream (", max_samples_per_stream,
                     ") must be ", kUnlimitedMaxSamples, " or >= 1"));
  }
  if (rate_limiter_timeout < absl::ZeroDuration()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "rate_limiter_timeout (", absl::FormatDuration(rate_limiter_timeout),
        ") must not be negative."));
  }
  return absl::OkStatus();
}

}  // namespace reverb
}  // namespace deepmind
