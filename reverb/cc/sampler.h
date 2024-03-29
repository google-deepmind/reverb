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

#ifndef LEARNING_DEEPMIND_REPLAY_REVERB_SAMPLER_H_
#define LEARNING_DEEPMIND_REPLAY_REVERB_SAMPLER_H_

#include <stddef.h>

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/queue.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/table.h"
#include "tensorflow/core/framework/tensor.h"

namespace deepmind {
namespace reverb {

// Converts `duration` to its millisecond equivalent.  If `duration` is
// `InfiniteDuration()`, then returns `-1`.
inline int64_t NonnegativeDurationToInt64Millis(absl::Duration duration) {
  return (duration == absl::InfiniteDuration())
             ? -1
             : absl::ToInt64Milliseconds(duration);
}

// The reverse operation to `NonnegativeDurationToInt64Millis`.
inline absl::Duration Int64MillisToNonnegativeDuration(int64_t milliseconds) {
  return (milliseconds < 0) ? absl::InfiniteDuration()
                            : absl::Milliseconds(milliseconds);
}

// A sample from the replay buffer.
class Sample {
 public:
  Sample(std::shared_ptr<const SampleInfo> info,
         std::vector<std::vector<tensorflow::Tensor>> column_chunks,
         std::vector<bool> squeeze_columns);

  // Returns the next time step from this sample as a flat sequence of tensors.
  // CHECK-fails if the entire sample has already been returned.
  std::vector<tensorflow::Tensor> GetNextTimestep();

  // Returns the trajectory as a flat sequence of tensors representing the
  // columns of the flattened trajectory.
  //
  // Fails with `DataLossError` if `GetNextTimestep()` has already been called
  // on this sample.
  absl::Status AsTrajectory(std::vector<tensorflow::Tensor>* data);

  // Returns true if the end of the sample has been reached.
  ABSL_MUST_USE_RESULT bool is_end_of_sample() const;

  // Returns true if the sample can be decomposed into timesteps.
  ABSL_MUST_USE_RESULT bool is_composed_of_timesteps() const;

  // Metadata info for the sampled item.
  //
  // NOTE! `info.item` may not be fully populated. Only `key`, `priority` and
  // `times_sampled` are guaranteed to be set.
  std::shared_ptr<const SampleInfo> info() const { return info_; }

 private:
  std::shared_ptr<const SampleInfo> info_;

  // Total number of time steps in this sample. Only set when
  // `is_timestep_sample()` is true.
  int64_t num_timesteps_;

  struct ColumnChunk {
    // Unpacked chunk, and potentially sliced, chunk content.
    tensorflow::Tensor tensor;

    // Index of the next sub slice to return when emitting timesteps.
    int offset = 0;
  };

  // Flat trajectory data. Each column uses a deque so the batched tensor can be
  // deallocated as soon as all its content has been emitted through
  // `GetNextTimestep`. If the data is retrieved with `AsBatchedTimesteps` or
  // `AsTrajectory` then the column chunks are concatenated instead of
  // subsliced.
  std::vector<std::deque<ColumnChunk>> columns_;

  // Columns where the batch dimension should be emitted. This is only respected
  // by `AsTrajectory`.
  std::vector<bool> squeeze_columns_;

  // The next time step to return when GetNextTimestep() is called.
  int64_t next_timestep_index_;

  // True if GetNextTimestep() has been called on this sample.
  bool next_timestep_called_;
};

// SamplerWorker implements strategy for fetching samples from table.
class SamplerWorker {
 public:
  virtual ~SamplerWorker() = default;

  // When called, future calls to FetchSampes must return CancelledError.
  virtual void Cancel() = 0;

  // Attempt to sample up to `num_samples` and push results to `queue`. Returns
  // when `num_samples` pushed to `queue` or error encountered.
  virtual std::pair<int64_t, absl::Status> FetchSamples(
      internal::Queue<std::unique_ptr<Sample>>* queue, int64_t num_samples,
      absl::Duration rate_limiter_timeout) = 0;
};

// The `Sampler` class should be used to retrieve samples from a
// ReverbService. A set of workers, each managing a  bi-directional gRPC stream
// are created. The workers unpack the responses into samples (sequences of
// timesteps) which are returned through calls to `GetNextTimestep` and
// `GetNextTrajectory`.
//
// Concurrent calls to `GetNextTimestep` is NOT supported! This includes calling
// `GetNextTrajectory` and `GetNextTimestep` concurrently.
//
// Terminology:
//   Timestep:
//      Set of tensors representing a single "step" (i.e data passed to
//      `Writer::Append`).
//   Chunk:
//      Timesteps batched (along the time dimension) and compressed. If each
//      timestep contains K tensors of dtype dt_k and shape s_k and the chunk
//      has length N then the chunk will contain K (compressed) tensors of dtype
//      dt_k and shape [N, ...s_k].
//   Sample:
//      Metadata (i.e priority, key) and sequence of timesteps that constitutes
//      an item in a `Table`. During transmission the "sample" is made
//      up of a vector of chunks and a metadata that defines what parts of the
//      chunks are actually part of the sample. Once received the sample is
//      unpacked into a sequence of `Timestep` before being returned to caller.
//   Worker:
//     Instance of `Sampler::Worker` running within its own thread managed
//     by the parent `Sampler`. The worker opens and manages
//     bi-directional gRPC streams to the server. It unpacks responses into
//     samples and pushes these into a `Queue` owned by the `Sampler`
//     (effectively merging the outputs of the workers).
//
// Each `Sampler` will create a set of `Worker`s, each managing a stream
// to a server. The combined output of the workers are merged into a `Queue` of
// complete samples. If `GetNextTimestep` is called then a sample is popped from
// the queue and split into timesteps and the first one returned. Timesteps are
// then popped one by one until the sample has been completely emitted and the
// process starts over. Calls to `GetNextTrajectory` skips the timestep
// splitting and returns samples as a "batch of timesteps".
//
class Sampler {
 public:
  static const int64_t kUnlimitedMaxSamples = -1;
  static const int kAutoSelectValue = -1;

  // By default, streams are only allowed to be open for a small number
  // (10000) of samples. A larger value could provide better performance
  // (reconnecting less frequently) but increases the risk of subtle "bias" in
  // the sampling distribution across a multi server setup. The bias will be
  // caused by a non uniform number of SampleStream-connections across the
  // servers being maintained for a longer period. The same phenomenon is
  // present with more short lived connections but is mitigated by the round
  // robin of the (more) frequently created new connections.
  // TODO(b/147425281): Set this value higher for localhost connections.
  static const int kDefaultMaxSamplesPerStream = 10000;

  // By default, only one worker is used as any higher number could lead to
  // incorrect behavior for FIFO samplers.
  static const int kDefaultNumWorkers = 1;

  // ID, probability, table size, priority, times sampled;
  static const int kNumInfoTensors = 5;

  // Extracts the `kNumInfoTensors` from `info` as scalar tensors and prepends
  // these fo `data`.
  //
  // The metadata fields extracted are:
  //
  //   * [uint64_t] Key of the sampled item.
  //   * [double] The probability of selecting this item at the time it was
  //     sampled.
  //   * [int64_t] The size of the table when the item was sampled.
  //   * [double] The priority of the item when it was sampled.
  //   * [int32_t] The number of times the item has been sampled (including this
  //     sample).
  //
  static std::vector<tensorflow::Tensor> WithInfoTensors(
      const SampleInfo& info, std::vector<tensorflow::Tensor> data);

  struct Options {
    // `max_samples` is the maximum number of samples the object will return.
    // Must be a positive number or `kUnlimitedMaxSamples`.
    int64_t max_samples = kUnlimitedMaxSamples;

    // `max_in_flight_samples_per_worker` is the number of samples requested by
    // a worker in each batch. A new batch is requested once all the requested
    // samples have been received.
    int max_in_flight_samples_per_worker = 100;

    // `num_workers` is the number of worker threads started.
    //
    // When set to `kAutoSelectValue`, `kDefaultNumWorkers` is used.
    int num_workers = kAutoSelectValue;

    // `max_samples_per_stream` is the maximum number of samples to fetch from a
    // stream before a new call is made. Keeping this number low ensures that
    // the data is fetched uniformly from all servers behind the `stub`.
    //
    // When set to `kAutoSelectValue`, `kDefaultMaxSamplesPerStream` is used.
    int max_samples_per_stream = kAutoSelectValue;

    // `rate_limiter_timeout` is the timeout that workers will use when waiting
    // for samples on a stream. This timeout is passed directly to the
    // `Table::Sample()` call on the server. When a timeout occurs, the Sample
    // status of `DeadlineExceeded` is returned.
    //
    // Note that if `num_workers > 1`, then any worker hitting the timeout will
    // lead to the Sampler returning a `DeadlineExceeded` in calls to
    // `GetNextTrajectory()` and/or `GetNextTimestep()`.
    //
    // The default is to wait forever - or until the connection closes, or
    // `Close` is called, whichever comes first.
    absl::Duration rate_limiter_timeout = absl::InfiniteDuration();

    // Checks that field values are valid and returns `InvalidArgument` if any
    // field value invalid.
    absl::Status Validate() const;
  };

  // Constructs a new `Sampler` using gRPC streams.
  //
  // `stub` is a connected gRPC stub to the ReverbService.
  // `table_name` is the name of the `Table` to sample from.
  // `options` defines details of how to samples.
  // `dtypes_and_shapes` describes the output signature (if any) to expect.
  Sampler(std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
          const std::string& table_name, const Options& options,
          internal::DtypesAndShapes dtypes_and_shapes = absl::nullopt);

  // Constructs a new `Sampler` which samples directly from local `table`.
  //
  // `table` is the table to sample from.
  // `options` defines details of how to samples.
  // `dtypes_and_shapes` describes the output signature (if any) to expect.
  Sampler(std::shared_ptr<Table> table, const Options& options,
          internal::DtypesAndShapes dtypes_and_shapes = absl::nullopt);

  // Joins worker threads through call to `Close`.
  ~Sampler();

  // Blocks until a timestep has been retrieved or until a non transient error
  // is encountered or `Close` has been called.
  //
  // `data` is populated with the next timestep of the sampled item. When the
  // the last timestep of the item is returned `end_of_sequence` is set to
  // `true`.
  //
  // NOTE! `info.item` may not be fully populated. Only `key`, `priority` and
  // `times_sampled` are guaranteed to be set.
  absl::Status GetNextTimestep(
      std::vector<tensorflow::Tensor>* data, bool* end_of_sequence,
      std::shared_ptr<const SampleInfo>* info = nullptr);

  // Blocks until a complete sample has been retrieved or until a non transient
  // error is encountered or `Close` has been called.
  //
  // `data` is populated with the full (flattened) trajectory represented by
  // the sampled item.
  //
  // NOTE! `info.item` may not be fully populated. Only `key`, `priority` and
  // `times_sampled` are guaranteed to be set.
  absl::Status GetNextTrajectory(
      std::vector<tensorflow::Tensor>* data,
      std::shared_ptr<const SampleInfo>* info = nullptr);

  // Cancels all workers and joins their threads. Any blocking or future call
  // to `GetNextTimestep` or `GetNextTrajectory` will return CancelledError
  // without blocking.
  void Close();

  // Sampler is neither copyable nor movable.
  Sampler(const Sampler&) = delete;
  Sampler& operator=(const Sampler&) = delete;

 private:
  Sampler(std::vector<std::unique_ptr<SamplerWorker>>, const std::string& table,
          const Options& options, internal::DtypesAndShapes dtypes_and_shapes);

  absl::Status ValidateAgainstOutputSpec(
      const std::vector<tensorflow::Tensor>& data);

  void RunWorker(SamplerWorker* worker) ABSL_LOCKS_EXCLUDED(mu_);

  // If `active_sample_` has been read, blocks until a sample has been retrieved
  // (popped from `samples_`) and populates `active_sample_`.
  absl::Status MaybeSampleNext();

  // Blocks until a complete sample has been retrieved or until a non transient
  // error is encountered or `Close` has been called. Note that this method does
  // NOT increment `returned_`. This is left to `GetNextTimestep` and
  // `GetNextTrajectory`. The returned pointer is only valid if the status is
  // OK.
  absl::Status PopNextSample(std::unique_ptr<Sample>* sample);

  // True if the workers should be shut down. This is the case when either:
  //  - `Close` has been called.
  //  - The number of returned samples equal `max_samples_`.
  //  - One of the worker streams has been closed with a non transient error
  //  status.
  bool should_stop_workers() const ABSL_SHARED_LOCKS_REQUIRED(mu_);

  // Stub used by workers to open SampleStream-connections to the servers. Note
  // that the endpoints are load balanced using "roundrobin" which results in
  // uniform sampling when using multiple backends.
  std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub_;

  // Name of the table we are sampling from.
  std::string table_;

  // The maximum number of samples to fetch. Calls to `GetNextTimestep` or
  // `GetNextTrajectory` after `max_samples_` has been returned will result in
  // OutOfRangeError.
  const int64_t max_samples_;

  // The maximum number of samples to stream from a single call. Once the number
  // of samples has been reached, a new stream is opened through the `stub_`.
  // This ensures that data is fetched from all the servers.
  const int64_t max_samples_per_stream_;

  // The rate limiter timeout argument that all workers pass to SampleStream.
  const absl::Duration rate_limiter_timeout_;

  // The number of complete samples that have been successfully requested.
  int64_t requested_ ABSL_GUARDED_BY(mu_) = 0;

  // The number of complete samples that have been returned through
  // `GetNextTimestep`.
  int64_t returned_ ABSL_GUARDED_BY(mu_) = 0;

  // Workers and threads managing the worker with the same index.
  std::vector<std::unique_ptr<SamplerWorker>> workers_;
  std::vector<std::unique_ptr<internal::Thread>> worker_threads_;

  // OK or the first non transient error encountered by a worker.
  absl::Status worker_status_ ABSL_GUARDED_BY(mu_);

  // Remaining timesteps of the currently active sample. Not that this is not
  // protected by mutex as concurrent calls to `GetNextTimestep` is not
  // supported.
  std::unique_ptr<Sample> active_sample_;

  // Queue of complete samples (timesteps batched up by into sequence).
  internal::Queue<std::unique_ptr<Sample>> samples_;

  // The dtypes and shapes users expect from either `GetNextTimestep` or
  // `GetNextTrajectory` (whichever they plan to call).  May be absl::nullopt,
  // meaning unknown.
  const internal::DtypesAndShapes dtypes_and_shapes_;
  const internal::DtypesAndShapes dtypes_and_shapes_for_sequence_;

  // Set if `Close` called.
  bool closed_ ABSL_GUARDED_BY(mu_) = false;

  mutable absl::Mutex mu_;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // LEARNING_DEEPMIND_REPLAY_REVERB_SAMPLER_H_
