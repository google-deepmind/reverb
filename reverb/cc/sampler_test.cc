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
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "grpcpp/client_context.h"
#include "grpcpp/impl/codegen/call_op_set.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/support/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/grpc/include/grpcpp/client_context.h"
#include "third_party/grpc/include/grpcpp/impl/call_op_set.h"
#include "third_party/grpc/include/grpcpp/support/sync_stream.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/reverb_service_mock.grpc.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/table.h"
#include "reverb/cc/tensor_compression.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "reverb/cc/testing/tensor_testutil.h"
#include "reverb/cc/testing/time_testutil.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"

namespace deepmind {
namespace reverb {
namespace {

using test::ExpectTensorEqual;
using testing::MakeSequenceRange;
using ::testing::SizeIs;

class FakeStream
    : public grpc::ClientReaderWriterInterface<SampleStreamRequest,
                                               SampleStreamResponse> {
 public:
  FakeStream(std::function<void(const SampleStreamRequest&)> on_write,
             std::vector<SampleStreamResponse> responses, grpc::Status status)
      : on_write_(std::move(on_write)),
        responses_(std::move(responses)),
        status_(std::move(status)) {}

  bool Write(const SampleStreamRequest& request,
             grpc::WriteOptions options) override {
    on_write_(request);
    return status_.ok();
  }

  bool Read(SampleStreamResponse* response) override {
    if (!responses_.empty() && status_.ok()) {
      *response = responses_.front();
      responses_.erase(responses_.begin());
      return true;
    }
    return false;
  }

  void WaitForInitialMetadata() override {}

  bool WritesDone() override { return true; }

  bool NextMessageSize(uint32_t* sz) override {
    *sz = responses_.front().ByteSizeLong();
    return true;
  }

  grpc::Status Finish() override { return status_; }

 private:
  std::function<void(const SampleStreamRequest&)> on_write_;
  std::vector<SampleStreamResponse> responses_;
  grpc::Status status_;
};

class FakeStub : public /* grpc_gen:: */MockReverbServiceStub {
 public:
  grpc::ClientReaderWriterInterface<SampleStreamRequest, SampleStreamResponse>*
  SampleStreamRaw(grpc::ClientContext* context) override {
    absl::WriterMutexLock lock(mu_);
    if (!streams_.empty()) {
      FakeStream* stream = streams_.front().release();
      streams_.pop_front();
      return stream;
    }

    return new FakeStream(
        [this](const SampleStreamRequest& request) {
          absl::WriterMutexLock lock(mu_);
          requests_.push_back(request);
        },
        {}, grpc::Status::OK);
  }

  void AddStream(std::vector<SampleStreamResponse> responses,
                 grpc::Status status = grpc::Status::OK) {
    absl::WriterMutexLock lock(mu_);
    streams_.push_back(std::make_unique<FakeStream>(
        [this](const SampleStreamRequest& request) {
          absl::WriterMutexLock lock(mu_);
          requests_.push_back(request);
        },
        std::move(responses), std::move(status)));
  }

  std::vector<SampleStreamRequest> requests() const {
    absl::ReaderMutexLock lock(mu_);
    return requests_;
  }

 private:
  std::list<std::unique_ptr<FakeStream>> streams_ ABSL_GUARDED_BY(mu_);
  std::vector<SampleStreamRequest> requests_ ABSL_GUARDED_BY(mu_);
  mutable absl::Mutex mu_;
};

std::shared_ptr<FakeStub> MakeFlakyStub(
    std::vector<SampleStreamResponse> responses,
    std::vector<grpc::Status> errors) {
  auto stub = std::make_shared<FakeStub>();
  for (const auto& error : errors) {
    stub->AddStream(responses, error);
  }
  stub->AddStream(responses);
  return stub;
}

std::shared_ptr<FakeStub> MakeGoodStub(
    std::vector<SampleStreamResponse> responses) {
  return MakeFlakyStub(std::move(responses), /*errors=*/{});
}

tensorflow::Tensor MakeTensor(int length) {
  tensorflow::TensorShape shape({length, 2});
  tensorflow::Tensor tensor(tensorflow::DT_UINT64, shape);
  for (int i = 0; i < tensor.NumElements(); i++) {
    tensor.flat<tensorflow::uint64>().data()[i] = i;
  }
  return tensor;
}

template <tensorflow::DataType dtype>
tensorflow::Tensor MakeConstantTensor(
    const tensorflow::TensorShape& shape,
    typename tensorflow::EnumToDataType<dtype>::Type value) {
  tensorflow::Tensor tensor(dtype, shape);
  for (int i = 0; i < tensor.NumElements(); i++) {
    tensor.flat<typename tensorflow::EnumToDataType<dtype>::Type>().data()[i] =
        value;
  }
  return tensor;
}

SampleStreamResponse MakeResponse(int item_length, bool delta_encode = false,
                                  int offset = 0, int data_length = 0,
                                  bool squeeze = false) {
  REVERB_CHECK(!squeeze || item_length == 1);

  if (data_length == 0) {
    data_length = item_length;
  }
  REVERB_CHECK_LE(item_length + offset, data_length);

  SampleStreamResponse response;
  auto* column = response.add_entries()->mutable_info()
                     ->mutable_item()
                     ->mutable_flat_trajectory()
                     ->add_columns();
  column->set_squeeze(squeeze);

  auto* slice = column->add_chunk_slices();
  slice->set_length(item_length);
  slice->set_offset(offset);

  auto tensor = MakeTensor(data_length);
  auto* chunk_data = response.mutable_entries(0)->add_data();
  response.mutable_entries(0)->set_end_of_sequence(true);
  if (delta_encode) {
    tensor = DeltaEncode(tensor, true);
    chunk_data->set_delta_encoded(true);
  }

  CHECK_OK(
      CompressTensorAsProto(tensor, chunk_data->mutable_data()->add_tensors()));

  chunk_data->mutable_sequence_range()->set_start(0);
  chunk_data->mutable_sequence_range()->set_end(data_length);

  return response;
}

std::shared_ptr<Table> MakeTable(int max_size = 100) {
  return std::make_shared<Table>(
      /*name=*/"queue",
      /*sampler=*/std::make_shared<FifoSelector>(),
      /*remover=*/std::make_shared<FifoSelector>(),
      /*max_size=*/max_size,
      /*max_times_sampled=*/1,
      /*rate_limiter=*/std::make_shared<RateLimiter>(1, 1, 0, max_size));
}

ChunkData MakeChunkData(uint64_t key, SequenceRange range) {
  ChunkData chunk;
  chunk.set_chunk_key(key);
  auto t = MakeTensor(range.end() - range.start() + 1);
  CHECK_OK(CompressTensorAsProto(t, chunk.mutable_data()->add_tensors()));
  *chunk.mutable_sequence_range() = std::move(range);

  return chunk;
}

TableItem MakeItem(uint64_t key, double priority,
                   const std::vector<SequenceRange>& sequences, int32_t offset,
                   int32_t length) {
  std::vector<std::shared_ptr<ChunkStore::Chunk>> chunks;
  std::vector<ChunkData> data(sequences.size());
  for (int i = 0; i < sequences.size(); i++) {
    data[i] = MakeChunkData(key * 100 + i, sequences[i]);
    chunks.push_back(std::make_shared<ChunkStore::Chunk>(data[i]));
  }

  Table::Item item(testing::MakePrioritizedItem(key, priority, data),
                   std::move(chunks));

  int32_t remaining = length;
  for (int slice_index = 0; slice_index < sequences.size(); slice_index++) {
    for (int col_index = 0; col_index < item.flat_trajectory().columns_size();
         col_index++) {
      auto* col =
          item.unsafe_mutable_flat_trajectory()->mutable_columns(col_index);
      auto* slice = col->mutable_chunk_slices(slice_index);
      slice->set_offset(offset);
      slice->set_length(
          std::min<int32_t>(slice->length() - slice->offset(), remaining));
      slice->set_index(col_index);
    }

    remaining -=
        item.flat_trajectory().columns(0).chunk_slices(slice_index).length();
    offset = 0;
  }

  return item;
}

void InsertItem(Table* table, uint64_t key, double priority,
                std::vector<int> sequence_lengths, int32_t offset = 0,
                int32_t length = 0, bool squeeze = false) {
  REVERB_CHECK(!squeeze || length == 1);
  if (length == 0) {
    length =
        std::accumulate(sequence_lengths.begin(), sequence_lengths.end(), 0) -
        offset;
  }

  std::vector<SequenceRange> ranges(sequence_lengths.size());
  int step_index = 0;
  for (int i = 0; i < sequence_lengths.size(); i++) {
    ranges[i] = MakeSequenceRange(100 * key, step_index,
                                  step_index + sequence_lengths[i] - 1);
    step_index += sequence_lengths[i];
  }

  auto item = MakeItem(key, priority, ranges, offset, length);
  item.unsafe_mutable_flat_trajectory()->mutable_columns(0)->set_squeeze(
      squeeze);
  REVERB_EXPECT_OK(table->InsertOrAssign(std::move(item)));
}

TEST(SampleTest, IsComposedOfTimesteps) {
  Sample timestep_sample(
      /*info=*/std::make_shared<SampleInfo>(),
      /*column_chunks=*/{{MakeTensor(5)}, {MakeTensor(5)}},
      /*squeeze_columns=*/{false});
  EXPECT_TRUE(timestep_sample.is_composed_of_timesteps());

  Sample non_timestep_sample(
      /*info=*/std::make_shared<SampleInfo>(),
      /*column_chunks=*/{{MakeTensor(5)}, {MakeTensor(10)}},
      /*squeeze_columns=*/{false});
  EXPECT_FALSE(non_timestep_sample.is_composed_of_timesteps());
}

TEST(GrpcSamplerTest, SendsFirstRequest) {
  auto stub = MakeGoodStub({MakeResponse(1)});
  Sampler sampler(stub, "table", {1, 1, 1});
  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;
  REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  EXPECT_THAT(stub->requests(), SizeIs(1));
}

TEST(GrpcSamplerTest, SetsEndOfSequence) {
  auto stub = MakeGoodStub({MakeResponse(2), MakeResponse(1)});
  Sampler sampler(stub, "table", {2, 1});

  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;

  // First sequence has 2 timesteps so first timestep should not be the end of
  // a sequence.
  REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  absl::SleepFor(absl::Milliseconds(5));
  EXPECT_FALSE(end_of_sequence);

  // Second timestep is the end of the first sequence.
  REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  absl::SleepFor(absl::Milliseconds(5));
  EXPECT_TRUE(end_of_sequence);

  // Third timestep is the first and only timestep of the second sequence.
  REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  absl::SleepFor(absl::Milliseconds(5));
  EXPECT_TRUE(end_of_sequence);
}

TEST(LocalSamplerTest, SetsEndOfSequence) {
  auto table = MakeTable();
  InsertItem(table.get(), 1, 1.0, {2});
  InsertItem(table.get(), 2, 1.0, {1});

  Sampler sampler(table, {2});

  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;

  // First sequence has 2 timesteps so first timestep should not be the end of
  // a sequence.
  REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  EXPECT_FALSE(end_of_sequence);

  // Second timestep is the end of the first sequence.
  REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  EXPECT_TRUE(end_of_sequence);

  // Third timestep is the first and only timestep of the second sequence.
  REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  EXPECT_TRUE(end_of_sequence);
}

TEST(GrpcSamplerTest, GetNextTrajectorySqueezesColumnsIfSet) {
  auto stub = MakeGoodStub({
      MakeResponse(
          /*item_length=*/1,
          /*delta_encode=*/false,
          /*offset=*/1,
          /*data_length=*/4,
          /*squeeze=*/true),
      MakeResponse(
          /*item_length=*/1,
          /*delta_encode=*/false,
          /*offset=*/1,
          /*data_length=*/4,
          /*squeeze=*/false),
  });
  Sampler sampler(stub, "table", {3, 1});

  std::vector<tensorflow::Tensor> squeezed;
  REVERB_EXPECT_OK(sampler.GetNextTrajectory(&squeezed));
  ASSERT_THAT(squeezed, SizeIs(1));
  ExpectTensorEqual<tensorflow::uint64>(
      squeezed[0], tensorflow::tensor::DeepCopy(MakeTensor(4).SubSlice(1)));

  std::vector<tensorflow::Tensor> not_squeezed;
  REVERB_EXPECT_OK(sampler.GetNextTrajectory(&not_squeezed));
  ASSERT_THAT(not_squeezed, SizeIs(1));
  ExpectTensorEqual<tensorflow::uint64>(
      not_squeezed[0], tensorflow::tensor::DeepCopy(MakeTensor(4).Slice(1, 2)));
}

TEST(LocalSamplerTest, GetNextTrajectorySqueezesColumnsIfSet) {
  auto table = MakeTable();
  InsertItem(
      /*table=*/table.get(),
      /*key=*/1,
      /*priority=*/1.0,
      /*sequence_lengths=*/{5},
      /*offset=*/2,
      /*length=*/1,
      /*squeeze=*/true);

  InsertItem(
      /*table=*/table.get(),
      /*key=*/2,
      /*priority=*/1.0,
      /*sequence_lengths=*/{5},
      /*offset=*/2,
      /*length=*/1,
      /*squeeze=*/false);

  Sampler sampler(table, {2});

  std::vector<tensorflow::Tensor> squeezed;
  REVERB_EXPECT_OK(sampler.GetNextTrajectory(&squeezed));
  ASSERT_THAT(squeezed, SizeIs(1));
  ExpectTensorEqual<tensorflow::uint64>(
      squeezed[0], tensorflow::tensor::DeepCopy(MakeTensor(4).SubSlice(2)));

  std::vector<tensorflow::Tensor> not_squeezed;
  REVERB_EXPECT_OK(sampler.GetNextTrajectory(&not_squeezed));
  ASSERT_THAT(not_squeezed, SizeIs(1));
  ExpectTensorEqual<tensorflow::uint64>(
      not_squeezed[0], tensorflow::tensor::DeepCopy(MakeTensor(4).Slice(2, 3)));
}

TEST(LocalSamplerTest, RespectsMaxInFlightItems) {
  auto table = MakeTable(100);
  for (int i = 0; i < 100; i++) {
    InsertItem(table.get(), i + 1, 1.0, {1});
  }

  Sampler::Options options;
  options.max_samples = 100;
  options.max_in_flight_samples_per_worker = 3;
  Sampler sampler(table, options);

  for (int i = 0; i < options.max_samples; i++) {
    int num_samples =
        table->info().rate_limiter_info().sample_stats().completed();
    int in_flight_items = num_samples - i;

    EXPECT_LE(in_flight_items, options.max_in_flight_samples_per_worker + 1);
    EXPECT_GE(in_flight_items, 0);

    std::vector<tensorflow::Tensor> sample;
    REVERB_ASSERT_OK(sampler.GetNextTrajectory(&sample));
  }
}

TEST(LocalSamplerTest, Close) {
  auto table = MakeTable();
  InsertItem(table.get(), 1, 1.0, {5});
  InsertItem(table.get(), 2, 1.0, {3});

  Sampler sampler(table, {3});

  std::vector<tensorflow::Tensor> first;
  REVERB_EXPECT_OK(sampler.GetNextTrajectory(&first));

  std::vector<tensorflow::Tensor> second;
  REVERB_EXPECT_OK(sampler.GetNextTrajectory(&second));

  sampler.Close();

  std::vector<tensorflow::Tensor> third;
  EXPECT_EQ(sampler.GetNextTrajectory(&third).code(),
            absl::StatusCode::kCancelled);
}

TEST(GrpcSamplerTest, RespectsBufferSizeAndMaxSamples) {
  const int kMaxSamples = 20;
  const int kMaxInFlightSamplesPerWorker = 11;
  const int kNumWorkers = 1;

  std::vector<SampleStreamResponse> responses;
  for (int i = 0; i < 40; i++) responses.push_back(MakeResponse(1));
  auto stub = MakeGoodStub(std::move(responses));

  Sampler sampler(stub, "table",
                  {kMaxSamples, kMaxInFlightSamplesPerWorker, kNumWorkers});

  test::WaitFor(
      [&]() {
        return !stub->requests().empty() && stub->requests()[0].num_samples() ==
                                                kMaxInFlightSamplesPerWorker;
      },
      absl::Milliseconds(10), 100);

  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;

  // The first request should aim to fill up the buffer.
  ASSERT_THAT(stub->requests(), SizeIs(1));
  EXPECT_EQ(stub->requests()[0].num_samples(), kMaxInFlightSamplesPerWorker);

  // Worker will issue another request when there is space for 9 elements in the
  // buffer. Retrieving 8 elements is not trigerring a request.
  for (int i = 0; i < 8; i++) {
    REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  }

  test::WaitFor([&]() { return stub->requests().size() == 1; },
                absl::Milliseconds(10), 100);
  EXPECT_THAT(stub->requests(), SizeIs(1));

  // Getting 9th elements allows worker to complete (kMaxSamples == 11 + 9).
  REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  test::WaitFor(
      [&]() {
        return stub->requests().size() == 2 &&
               stub->requests()[1].num_samples() ==
                   kMaxSamples - kMaxInFlightSamplesPerWorker;
      },
      absl::Milliseconds(10), 100);
  EXPECT_THAT(stub->requests(), SizeIs(2));

  // The second request should respect the `max_samples` and thus only request
  // 9 (9 + 11 = 20) more samples.
  EXPECT_EQ(stub->requests()[1].num_samples(),
            kMaxSamples - kMaxInFlightSamplesPerWorker);

  // Consuming the remaining 10 samples should not trigger any more requests
  // as this would violate `max_samples`.
  for (int i = 0; i < 10; i++) {
    REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  }
  test::WaitFor([&]() { return stub->requests().size() == 2; },
                absl::Milliseconds(10), 100);
  EXPECT_THAT(stub->requests(), SizeIs(2));
}

TEST(GrpcSamplerTest, UnpacksDeltaEncodedTensors) {
  auto stub = MakeGoodStub({MakeResponse(10, false), MakeResponse(10, true)});
  Sampler sampler(stub, "table", {2, 1});
  std::vector<tensorflow::Tensor> not_encoded;
  std::vector<tensorflow::Tensor> encoded;
  REVERB_EXPECT_OK(sampler.GetNextTrajectory(&not_encoded));
  REVERB_EXPECT_OK(sampler.GetNextTrajectory(&encoded));
  ASSERT_EQ(not_encoded.size(), encoded.size());
  EXPECT_EQ(encoded[0].dtype(), tensorflow::DT_UINT64);
  for (int i = 4; i < encoded.size(); i++) {
    ExpectTensorEqual<tensorflow::uint64>(encoded[i], not_encoded[i]);
  }
}

TEST(GrpcSamplerTest, GetNextTimestepForwardsFatalServerError) {
  const int kNumWorkers = 4;
  const int kItemLength = 10;
  const auto kError = grpc::Status(grpc::StatusCode::NOT_FOUND, "");

  auto stub = MakeFlakyStub({MakeResponse(kItemLength)}, {kError});
  Sampler sampler(stub, "table",
                  {Sampler::kUnlimitedMaxSamples, 1, kNumWorkers});

  // It is possible that the sample returned by one of the workers is reached
  // before the failing worker has reported it's error so we need to pop at
  // least two samples to ensure that the we will see the error.
  absl::Status status;
  for (int i = 0; status.ok() && i < kItemLength + 1; i++) {
    std::vector<tensorflow::Tensor> sample;
    bool end_of_sequence;
    status = sampler.GetNextTimestep(&sample, &end_of_sequence);
  }
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
  sampler.Close();
}

TEST(LocalSamplerTest, GetNextTrajectoryForwardsFatalServerError) {
  auto table = MakeTable();

  Sampler::Options options;
  options.rate_limiter_timeout = absl::Milliseconds(10);
  Sampler sampler(table, options);

  std::vector<tensorflow::Tensor> sample;
  auto status = sampler.GetNextTrajectory(&sample);
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  sampler.Close();
}

TEST(GrpcSamplerTest, GetNextTrajectoryForwardsFatalServerError) {
  const int kNumWorkers = 4;
  const int kItemLength = 10;
  const auto kError = grpc::Status(grpc::StatusCode::NOT_FOUND, "");

  auto stub = MakeFlakyStub({MakeResponse(kItemLength)}, {kError});
  Sampler sampler(stub, "table",
                  {Sampler::kUnlimitedMaxSamples, 1, kNumWorkers});

  // It is possible that the sample returned by one of the workers is reached
  // before the failing worker has reported it's error so we need to pop at
  // least two samples to ensure that the we will see the error.
  absl::Status status;
  for (int i = 0; status.ok() && i < 2; i++) {
    std::vector<tensorflow::Tensor> sample;
    status = sampler.GetNextTrajectory(&sample);
  }
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
}

TEST(LocalSamplerTest, GetNextTimestepForwardsFatalServerError) {
  auto table = MakeTable();

  Sampler::Options options;
  options.rate_limiter_timeout = absl::Milliseconds(10);
  Sampler sampler(table, options);

  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;
  auto status = sampler.GetNextTimestep(&sample, &end_of_sequence);
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  sampler.Close();
}

class ParameterizedGrpcSamplerTest : public ::testing::Test,
                public ::testing::WithParamInterface<grpc::StatusCode> {};

TEST_P(ParameterizedGrpcSamplerTest, GetNextTimestepRetriesTransientErrors) {
  const int kNumWorkers = 2;
  const int kItemLength = 10;
  const auto kError = grpc::Status(GetParam(), "");

  auto stub = MakeFlakyStub(
      {MakeResponse(kItemLength), MakeResponse(kItemLength)}, {kError});
  Sampler sampler(stub, "table",
                  {Sampler::kUnlimitedMaxSamples, 1, kNumWorkers});

  // It is possible that the sample returned by one of the workers is reached
  // before the failing worker has reported it's error so we need to pop at
  // least two samples to ensure that the we will see the error.
  for (int i = 0; i < kItemLength + 1; i++) {
    std::vector<tensorflow::Tensor> sample;
    bool end_of_sequence;
    REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  }
}

TEST_P(ParameterizedGrpcSamplerTest, GetNextTrajectoryRetriesTransientErrors) {
  const int kNumWorkers = 2;
  const int kItemLength = 10;
  const auto kError = grpc::Status(GetParam(), "");

  auto stub = MakeFlakyStub(
      {MakeResponse(kItemLength), MakeResponse(kItemLength)}, {kError});
  Sampler sampler(stub, "table",
                  {Sampler::kUnlimitedMaxSamples, 1, kNumWorkers});

  // It is possible that the sample returned by one of the workers is reached
  // before the failing worker has reported it's error so we need to pop at
  // least two samples to ensure that the we will see the error.
  for (int i = 0; i < 2; i++) {
    std::vector<tensorflow::Tensor> sample;
    REVERB_EXPECT_OK(sampler.GetNextTrajectory(&sample));
  }
}

INSTANTIATE_TEST_CASE_P(ErrorTests, ParameterizedGrpcSamplerTest,
                        ::testing::Values(grpc::StatusCode::UNAVAILABLE,
                                          grpc::StatusCode::CANCELLED));

TEST(GrpcSamplerTest, GetNextTimestepReturnsErrorIfMaximumSamplesExceeded) {
  auto stub = MakeGoodStub({MakeResponse(1), MakeResponse(1), MakeResponse(1)});
  Sampler sampler(stub, "table", {2, 1, 1});
  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;
  REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  EXPECT_EQ(sampler.GetNextTimestep(&sample, &end_of_sequence).code(),
            absl::StatusCode::kOutOfRange);
}

TEST(LocalSamplerTest, GetNextTimestepReturnsErrorIfMaximumSamplesExceeded) {
  auto table = MakeTable();
  InsertItem(table.get(), 1, 1.0, {1});
  InsertItem(table.get(), 2, 1.0, {1});
  InsertItem(table.get(), 3, 1.0, {1});

  Sampler sampler(table, {2});

  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;
  REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  EXPECT_EQ(sampler.GetNextTimestep(&sample, &end_of_sequence).code(),
            absl::StatusCode::kOutOfRange);
}

TEST(GrpcSamplerTest, GetNextTimestepReturnsErrorIfNotDecomposible) {
  auto response = MakeResponse(5);
  auto* entry = response.mutable_entries(0);

  // Add a column of length 10 to the existing one of length 5.
  ASSERT_OK(CompressTensorAsProto(
      MakeTensor(10), entry->add_data()->mutable_data()->add_tensors()));
  auto* slice = entry->mutable_info()
                    ->mutable_item()
                    ->mutable_flat_trajectory()
                    ->add_columns()
                    ->add_chunk_slices();
  *slice = entry->info().item().flat_trajectory().columns(0).chunk_slices(0);
  slice->set_index(1);
  slice->set_length(10);

  auto stub = MakeGoodStub({std::move(response)});
  Sampler sampler(stub, "table", {2, 1, 1});
  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;
  auto status = sampler.GetNextTimestep(&sample, &end_of_sequence);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
}

TEST(GrpcSamplerTest, GetNextTrajectoryReturnsErrorIfMaximumSamplesExceeded) {
  auto stub = MakeGoodStub({MakeResponse(5), MakeResponse(5), MakeResponse(5)});
  Sampler sampler(stub, "table", {2, 1, 1});
  std::vector<tensorflow::Tensor> sample;
  REVERB_EXPECT_OK(sampler.GetNextTrajectory(&sample));
  REVERB_EXPECT_OK(sampler.GetNextTrajectory(&sample));
  EXPECT_EQ(sampler.GetNextTrajectory(&sample).code(),
            absl::StatusCode::kOutOfRange);
}

TEST(LocalSamplerTest, GetNextTrajectoryReturnsErrorIfMaximumSamplesExceeded) {
  auto table = MakeTable();
  InsertItem(table.get(), 1, 1.0, {2});
  InsertItem(table.get(), 2, 1.0, {2});
  InsertItem(table.get(), 3, 1.0, {2});

  Sampler sampler(table, {2});

  std::vector<tensorflow::Tensor> sample;
  REVERB_EXPECT_OK(sampler.GetNextTrajectory(&sample));
  REVERB_EXPECT_OK(sampler.GetNextTrajectory(&sample));
  EXPECT_EQ(sampler.GetNextTrajectory(&sample).code(),
            absl::StatusCode::kOutOfRange);
}

TEST(GrpcSamplerTest, StressTestWithoutErrors) {
  const int kNumWorkers = 100;  // Should be larger than the number of CPUs.
  const int kMaxSamples = 10000;
  const int kMaxSamplesPerStream = 50;
  const int kMaxInflightSamplesPerStream = 7;
  const int kItemLength = 3;

  std::vector<SampleStreamResponse> responses(kMaxSamplesPerStream);
  for (int i = 0; i < kMaxSamplesPerStream; i++) {
    responses[i] = MakeResponse(kItemLength);
  }

  auto stub = std::make_shared<FakeStub>();
  for (int i = 0; i < (kMaxSamples / kMaxSamplesPerStream) + kNumWorkers; i++) {
    stub->AddStream(responses);
  }

  Sampler sampler(stub, "table",
                  {kMaxSamples, kMaxInflightSamplesPerStream, kNumWorkers,
                   kMaxSamplesPerStream});

  for (int i = 0; i < kItemLength * kMaxSamples; i++) {
    std::vector<tensorflow::Tensor> sample;
    bool end_of_sequence;
    REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  }

  // There should be no more samples left.
  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;
  EXPECT_EQ(sampler.GetNextTimestep(&sample, &end_of_sequence).code(),
            absl::StatusCode::kOutOfRange);
}

TEST(LocalSamplerTest, StressTestWithoutErrors) {
  const int kNumWorkers = 100;  // Should be larger than the number of CPUs.
  const int kMaxSamples = 10000;
  const int kItemLength = 3;

  auto table = MakeTable(kMaxSamples);
  for (int i = 0; i < kMaxSamples; i++) {
    InsertItem(table.get(), i, 1.0, {kItemLength});
  }

  Sampler::Options options;
  options.num_workers = kNumWorkers;
  options.max_samples = kMaxSamples;
  Sampler sampler(table, options);

  for (int i = 0; i < kItemLength * kMaxSamples; i++) {
    std::vector<tensorflow::Tensor> sample;
    bool end_of_sequence;
    REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  }

  // There should be no more samples left.
  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;
  EXPECT_EQ(sampler.GetNextTimestep(&sample, &end_of_sequence).code(),
            absl::StatusCode::kOutOfRange);
  sampler.Close();
}

TEST(GrpcSamplerTest, StressTestWithTransientErrors) {
  const int kNumWorkers = 100;  // Should be larger than the number of CPUs.
  const int kMaxSamples = 10000;
  const int kMaxSamplesPerStream = 50;
  const int kMaxInflightSamplesPerStream = 7;
  const int kItemLength = 3;
  const int kTransientErrorFrequency = 23;

  std::vector<SampleStreamResponse> responses(kMaxSamplesPerStream);
  for (int i = 0; i < kMaxSamplesPerStream; i++) {
    responses[i] = MakeResponse(kItemLength);
  }

  auto stub = std::make_shared<FakeStub>();
  for (int i = 0; i < (kMaxSamples / kMaxSamplesPerStream) + kNumWorkers; i++) {
    auto status = i % kTransientErrorFrequency != 0
                      ? grpc::Status::OK
                      : grpc::Status(grpc::StatusCode::UNAVAILABLE, "");
    stub->AddStream(responses, status);
  }

  Sampler sampler(stub, "table",
                  {kMaxSamples, kMaxInflightSamplesPerStream, kNumWorkers,
                   kMaxSamplesPerStream});

  for (int i = 0; i < kItemLength * kMaxSamples; i++) {
    std::vector<tensorflow::Tensor> sample;
    bool end_of_sequence;
    REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  }

  // There should be no more samples left.
  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;
  EXPECT_EQ(sampler.GetNextTimestep(&sample, &end_of_sequence).code(),
            absl::StatusCode::kOutOfRange);
}

TEST(SamplerDeathTest, DiesIfMaxInFlightSamplesPerWorkerIsNonPositive) {
  Sampler::Options options;

  options.max_in_flight_samples_per_worker = 0;
  ASSERT_DEATH(Sampler sampler(MakeGoodStub({}), "table", options), "");
  ASSERT_DEATH(Sampler sampler(nullptr, options), "");

  options.max_in_flight_samples_per_worker = -1;
  ASSERT_DEATH(Sampler sampler(MakeGoodStub({}), "table", options), "");
  ASSERT_DEATH(Sampler sampler(nullptr, options), "");
}

TEST(SamplerDeathTest, DiesIfMaxSamplesInvalid) {
  Sampler::Options options;

  options.max_samples = -2;
  ASSERT_DEATH(Sampler sampler(MakeGoodStub({}), "table", options), "");
  ASSERT_DEATH(Sampler sampler(nullptr, options), "");

  options.max_samples = 0;
  ASSERT_DEATH(Sampler sampler(MakeGoodStub({}), "table", options), "");
  ASSERT_DEATH(Sampler sampler(nullptr, options), "");
}

TEST(SamplerDeathTest, DiesIfNumWorkersIsInvalid) {
  Sampler::Options options;

  options.num_workers = 0;
  ASSERT_DEATH(Sampler sampler(MakeGoodStub({}), "table", options), "");
  ASSERT_DEATH(Sampler sampler(nullptr, options), "");

  options.num_workers = -2;
  ASSERT_DEATH(Sampler sampler(MakeGoodStub({}), "table", options), "");
  ASSERT_DEATH(Sampler sampler(nullptr, options), "");
}

TEST(SamplerOptionsTest, ValidateDefaultOptions) {
  Sampler::Options options;
  REVERB_EXPECT_OK(options.Validate());
}

TEST(SamplerOptionsTest, ValidateChecksMaxSamples) {
  Sampler::Options options;
  options.max_samples = 0;
  EXPECT_EQ(options.Validate().code(), absl::StatusCode::kInvalidArgument);
  options.max_samples = Sampler::kUnlimitedMaxSamples;
  REVERB_EXPECT_OK(options.Validate());
  options.max_samples = -2;
  EXPECT_EQ(options.Validate().code(), absl::StatusCode::kInvalidArgument);
}

TEST(SamplerOptionsTest, ValidateChecksMaxInFlightSamplesPerWorker) {
  Sampler::Options options;
  options.max_in_flight_samples_per_worker = 0;
  EXPECT_EQ(options.Validate().code(), absl::StatusCode::kInvalidArgument);
  options.max_in_flight_samples_per_worker = -2;
  EXPECT_EQ(options.Validate().code(), absl::StatusCode::kInvalidArgument);
}

TEST(SamplerOptionsTest, ValidateChecksNumWorkers) {
  Sampler::Options options;
  options.num_workers = 0;
  EXPECT_EQ(options.Validate().code(), absl::StatusCode::kInvalidArgument);
  options.num_workers = Sampler::kAutoSelectValue;
  REVERB_EXPECT_OK(options.Validate());
  options.num_workers = -2;
  EXPECT_EQ(options.Validate().code(), absl::StatusCode::kInvalidArgument);
}

TEST(SamplerOptionsTest, ValidateChecksMaxSamplesPerStream) {
  Sampler::Options options;
  options.max_samples_per_stream = 0;
  EXPECT_EQ(options.Validate().code(), absl::StatusCode::kInvalidArgument);
  options.max_samples_per_stream = Sampler::kAutoSelectValue;
  REVERB_EXPECT_OK(options.Validate());
  options.max_samples_per_stream = -2;
  EXPECT_EQ(options.Validate().code(), absl::StatusCode::kInvalidArgument);
}

TEST(SamplerOptionsTest, ValidateChecksRateLimiterTimeout) {
  Sampler::Options options;
  options.rate_limiter_timeout = -absl::Seconds(1);
  EXPECT_EQ(options.Validate().code(), absl::StatusCode::kInvalidArgument);
  options.rate_limiter_timeout = absl::ZeroDuration();
  REVERB_EXPECT_OK(options.Validate());
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
