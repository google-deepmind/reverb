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

#include <list>
#include <vector>

#include "grpcpp/client_context.h"
#include "grpcpp/impl/codegen/call_op_set.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/support/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/reverb_service_mock.grpc.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/support/tf_util.h"
#include "reverb/cc/tensor_compression.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "reverb/cc/testing/tensor_testutil.h"
#include "reverb/cc/testing/time_testutil.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.pb.h"

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
      *response->mutable_data() = responses_.front().data();
      *response->mutable_info() = responses_.front().info();
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
    absl::WriterMutexLock lock(&mu_);
    if (!streams_.empty()) {
      FakeStream* stream = streams_.front().release();
      streams_.pop_front();
      return stream;
    }

    return new FakeStream(
        [this](const SampleStreamRequest& request) {
          absl::WriterMutexLock lock(&mu_);
          requests_.push_back(request);
        },
        {}, grpc::Status::OK);
  }

  void AddStream(std::vector<SampleStreamResponse> responses,
                 grpc::Status status = grpc::Status::OK) {
    absl::WriterMutexLock lock(&mu_);
    streams_.push_back(absl::make_unique<FakeStream>(
        [this](const SampleStreamRequest& request) {
          absl::WriterMutexLock lock(&mu_);
          requests_.push_back(request);
        },
        std::move(responses), std::move(status)));
  }

  std::vector<SampleStreamRequest> requests() const {
    absl::ReaderMutexLock lock(&mu_);
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
  auto* column = response.mutable_info()
                     ->mutable_item()
                     ->mutable_flat_trajectory()
                     ->add_columns();
  column->set_squeeze(squeeze);

  auto* slice = column->add_chunk_slices();
  slice->set_length(item_length);
  slice->set_offset(offset);

  auto tensor = MakeTensor(data_length);
  auto* chunk_data = response.add_data();
  if (delta_encode) {
    tensor = DeltaEncode(tensor, true);
    chunk_data->set_delta_encoded(true);
  }

  CompressTensorAsProto(tensor, chunk_data->mutable_data()->add_tensors());

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
  CompressTensorAsProto(t, chunk.mutable_data()->add_tensors());
  *chunk.mutable_sequence_range() = std::move(range);

  return chunk;
}

TableItem MakeItem(uint64_t key, double priority,
                   const std::vector<SequenceRange>& sequences, int32_t offset,
                   int32_t length) {
  TableItem item;

  std::vector<ChunkData> data(sequences.size());
  for (int i = 0; i < sequences.size(); i++) {
    data[i] = MakeChunkData(key * 100 + i, sequences[i]);
    item.chunks.push_back(std::make_shared<ChunkStore::Chunk>(data[i]));
  }

  item.item = testing::MakePrioritizedItem(key, priority, data);

  int32_t remaining = length;
  for (int slice_index = 0; slice_index < sequences.size(); slice_index++) {
    for (int col_index = 0;
         col_index < item.item.flat_trajectory().columns_size(); col_index++) {
      auto* col =
          item.item.mutable_flat_trajectory()->mutable_columns(col_index);
      auto* slice = col->mutable_chunk_slices(slice_index);
      slice->set_offset(offset);
      slice->set_length(
          std::min<int32_t>(slice->length() - slice->offset(), remaining));
      slice->set_index(col_index);
    }

    remaining -= item.item.flat_trajectory()
                     .columns(0)
                     .chunk_slices(slice_index)
                     .length();
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
  item.item.mutable_flat_trajectory()->mutable_columns(0)->set_squeeze(squeeze);
  REVERB_EXPECT_OK(table->InsertOrAssign(std::move(item)));
}

TEST(SampleTest, IsComposedOfTimesteps) {
  Sample timestep_sample(
      /*key=*/100,
      /*probability=*/0.5,
      /*table_size=*/2,
      /*priority=*/1,
      /*rate_limited=*/false,
      /*column_chunks=*/{{MakeTensor(5)}, {MakeTensor(5)}},
      /*squeeze_columns=*/{false});
  EXPECT_TRUE(timestep_sample.is_composed_of_timesteps());

  Sample non_timestep_sample(
      /*key=*/100,
      /*probability=*/0.5,
      /*table_size=*/2,
      /*priority=*/1,
      /*rate_limited=*/false,
      /*column_chunks=*/{{MakeTensor(5)}, {MakeTensor(10)}},
      /*squeeze_columns=*/{false});
  EXPECT_FALSE(non_timestep_sample.is_composed_of_timesteps());
}

TEST(SampleTest, RateLimited) {
  for (bool rate_limited : {true, false}) {
    Sample sample(
        /*key=*/100,
        /*probability=*/0.5,
        /*table_size=*/2,
        /*priority=*/1,
        /*rate_limited=*/rate_limited,
        /*column_chunks=*/{{MakeTensor(5)}, {MakeTensor(5)}},
        /*squeeze_columns=*/{false});
    EXPECT_EQ(sample.rate_limited(), rate_limited);
  }
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

TEST(GrpcSamplerTest, GetNextSampleReturnsPriority) {
  std::vector<SampleStreamResponse> responses = {MakeResponse(5),
                                                 MakeResponse(3)};
  responses[0].mutable_info()->mutable_item()->set_priority(100.0);
  responses[1].mutable_info()->mutable_item()->set_priority(101.0);
  auto stub = MakeGoodStub(responses);
  Sampler sampler(stub, "table", {2, 1});

  std::vector<tensorflow::Tensor> first;
  REVERB_EXPECT_OK(sampler.GetNextSample(&first));
  EXPECT_THAT(first,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<double>(
      first[3], MakeConstantTensor<tensorflow::DT_DOUBLE>({5}, 100.0));

  std::vector<tensorflow::Tensor> second;
  REVERB_EXPECT_OK(sampler.GetNextSample(&second));
  EXPECT_THAT(second,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<double>(
      second[3], MakeConstantTensor<tensorflow::DT_DOUBLE>({3}, 101.0));
}

TEST(LocalSamplerTest, GetNextSampleReturnsPriority) {
  auto table = MakeTable();
  InsertItem(table.get(), 1, 100.0, {5});
  InsertItem(table.get(), 2, 101.0, {3});

  Sampler::Options options;
  options.max_samples = 2;
  Sampler sampler(table, options);

  std::vector<tensorflow::Tensor> first;
  REVERB_EXPECT_OK(sampler.GetNextSample(&first));
  EXPECT_THAT(first,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<double>(
      first[3], MakeConstantTensor<tensorflow::DT_DOUBLE>({5}, 100.0));

  std::vector<tensorflow::Tensor> second;
  REVERB_EXPECT_OK(sampler.GetNextSample(&second));
  EXPECT_THAT(second,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<double>(
      second[3], MakeConstantTensor<tensorflow::DT_DOUBLE>({3}, 101.0));
}

TEST(GrpcSamplerTest, GetNextSampleReturnsWholeSequence) {
  auto stub = MakeGoodStub({MakeResponse(5), MakeResponse(3)});
  Sampler sampler(stub, "table", {2, 1});

  std::vector<tensorflow::Tensor> first;
  REVERB_EXPECT_OK(sampler.GetNextSample(&first));
  EXPECT_THAT(first,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<tensorflow::uint64>(first[4], MakeTensor(5));

  std::vector<tensorflow::Tensor> second;
  REVERB_EXPECT_OK(sampler.GetNextSample(&second));
  EXPECT_THAT(second,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<tensorflow::uint64>(second[4], MakeTensor(3));
}

TEST(LocalSamplerTest, GetNextSampleReturnsWholeSequence) {
  auto table = MakeTable();
  InsertItem(table.get(), 1, 1.0, {5});
  InsertItem(table.get(), 2, 1.0, {3});

  Sampler sampler(table, {2});

  std::vector<tensorflow::Tensor> first;
  REVERB_EXPECT_OK(sampler.GetNextSample(&first));
  EXPECT_THAT(first,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<tensorflow::uint64>(first[4], MakeTensor(5));

  std::vector<tensorflow::Tensor> second;
  REVERB_EXPECT_OK(sampler.GetNextSample(&second));
  EXPECT_THAT(second,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<tensorflow::uint64>(second[4], MakeTensor(3));
}

TEST(GrpcSamplerTest, GetNextSampleTrimsSequence) {
  auto stub = MakeGoodStub({
      MakeResponse(5, false, 1, 6),   // Trim offset at the start.
      MakeResponse(3, false, 0, 4),   // Trim timestep from end.
      MakeResponse(2, false, 1, 10),  // Trim offset and end.
  });
  Sampler sampler(stub, "table", {3, 1});

  std::vector<tensorflow::Tensor> start_trimmed;
  REVERB_EXPECT_OK(sampler.GetNextSample(&start_trimmed));
  ASSERT_THAT(start_trimmed,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<tensorflow::uint64>(
      start_trimmed[4],
      tensorflow::tensor::DeepCopy(MakeTensor(6).Slice(1, 6)));

  std::vector<tensorflow::Tensor> end_trimmed;
  REVERB_EXPECT_OK(sampler.GetNextSample(&end_trimmed));
  ASSERT_THAT(end_trimmed,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<tensorflow::uint64>(end_trimmed[4],
                                        MakeTensor(4).Slice(0, 3));

  std::vector<tensorflow::Tensor> start_and_end_trimmed;
  REVERB_EXPECT_OK(sampler.GetNextSample(&start_and_end_trimmed));
  ASSERT_THAT(start_and_end_trimmed,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<tensorflow::uint64>(
      start_and_end_trimmed[4],
      tensorflow::tensor::DeepCopy(MakeTensor(10).Slice(1, 3)));
}

TEST(LocalSamplerTest, GetNextSampleTrimsSequence) {
  auto table = MakeTable();
  InsertItem(table.get(), 1, 1.0, {5}, 1, 4);     // Trim offset at the start.
  InsertItem(table.get(), 2, 1.0, {3}, 0, 2);     // Trim at the end.
  InsertItem(table.get(), 3, 1.0, {2, 3}, 1, 2);  // Trim offset and end.

  Sampler sampler(table, {3});

  std::vector<tensorflow::Tensor> start_trimmed;
  REVERB_EXPECT_OK(sampler.GetNextSample(&start_trimmed));
  ASSERT_THAT(start_trimmed,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<tensorflow::uint64>(
      start_trimmed[4],
      tensorflow::tensor::DeepCopy(MakeTensor(5).Slice(1, 5)));

  std::vector<tensorflow::Tensor> end_trimmed;
  REVERB_EXPECT_OK(sampler.GetNextSample(&end_trimmed));
  ASSERT_THAT(end_trimmed,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<tensorflow::uint64>(end_trimmed[4],
                                        MakeTensor(3).Slice(0, 2));

  std::vector<tensorflow::Tensor> start_and_end_trimmed;
  REVERB_EXPECT_OK(sampler.GetNextSample(&start_and_end_trimmed));
  ASSERT_THAT(start_and_end_trimmed,
              SizeIs(5));  // ID, probability, table size, priority, data.

  tensorflow::Tensor start_and_end_trimmer_want;
  REVERB_EXPECT_OK(FromTensorflowStatus(tensorflow::tensor::Concat(
      {
          tensorflow::tensor::DeepCopy(MakeTensor(2).Slice(1, 2)),
          tensorflow::tensor::DeepCopy(MakeTensor(3).Slice(0, 1)),
      },
      &start_and_end_trimmer_want)));

  ExpectTensorEqual<tensorflow::uint64>(start_and_end_trimmed[4],
                                        start_and_end_trimmer_want);
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
  ASSERT_THAT(squeezed,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<tensorflow::uint64>(
      squeezed[4], tensorflow::tensor::DeepCopy(MakeTensor(4).SubSlice(1)));

  std::vector<tensorflow::Tensor> not_squeezed;
  REVERB_EXPECT_OK(sampler.GetNextTrajectory(&not_squeezed));
  ASSERT_THAT(not_squeezed,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<tensorflow::uint64>(
      not_squeezed[4], tensorflow::tensor::DeepCopy(MakeTensor(4).Slice(1, 2)));
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
  ASSERT_THAT(squeezed,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<tensorflow::uint64>(
      squeezed[4], tensorflow::tensor::DeepCopy(MakeTensor(4).SubSlice(2)));

  std::vector<tensorflow::Tensor> not_squeezed;
  REVERB_EXPECT_OK(sampler.GetNextTrajectory(&not_squeezed));
  ASSERT_THAT(not_squeezed,
              SizeIs(5));  // ID, probability, table size, priority, data.
  ExpectTensorEqual<tensorflow::uint64>(
      not_squeezed[4], tensorflow::tensor::DeepCopy(MakeTensor(4).Slice(2, 3)));
}

TEST(LocalSamplerTest, RespectsMaxInFlightItems) {
  auto table = MakeTable(100);
  for (int i = 0; i < 100; i++) {
    InsertItem(table.get(), i + 1, 1.0, {1});
  }

  Sampler::Options options;
  options.max_samples = 100;
  options.max_in_flight_samples_per_worker = 3;
  options.flexible_batch_size = 5;
  Sampler sampler(table, options);

  for (int i = 0; i < options.max_samples; i++) {
    int num_samples =
        table->info().rate_limiter_info().sample_stats().completed();
    int in_flight_items = num_samples - i;

    EXPECT_LE(in_flight_items, options.max_in_flight_samples_per_worker + 1);
    EXPECT_GE(in_flight_items, 0);

    std::vector<tensorflow::Tensor> sample;
    REVERB_ASSERT_OK(sampler.GetNextSample(&sample));
  }
}

TEST(LocalSamplerTest, Close) {
  auto table = MakeTable();
  InsertItem(table.get(), 1, 1.0, {5});
  InsertItem(table.get(), 2, 1.0, {3});

  Sampler sampler(table, {3});

  std::vector<tensorflow::Tensor> first;
  REVERB_EXPECT_OK(sampler.GetNextSample(&first));

  std::vector<tensorflow::Tensor> second;
  REVERB_EXPECT_OK(sampler.GetNextSample(&second));

  sampler.Close();

  std::vector<tensorflow::Tensor> third;
  EXPECT_EQ(sampler.GetNextSample(&third).code(), absl::StatusCode::kCancelled);
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

  // The queue outside the workers has size `num_workers` (i.e 1 here) so in
  // addition to the samples actually returned to the user, an additional
  // sample is considered to been "consumed" from the perspective of the worker.

  // The first 9 (9 + 1 = 10) pops should not result in a new request.
  for (int i = 0; i < kMaxInFlightSamplesPerWorker - kNumWorkers - 1; i++) {
    REVERB_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  }

  test::WaitFor([&]() { return stub->requests().size() == 1; },
                absl::Milliseconds(10), 100);
  EXPECT_THAT(stub->requests(), SizeIs(1));

  // The 10th sample (+1 in the queue) mean that all the requested samples
  // have been received and thus a new request is sent to retrieve more.
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
  REVERB_EXPECT_OK(sampler.GetNextSample(&not_encoded));
  REVERB_EXPECT_OK(sampler.GetNextSample(&encoded));
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

TEST(LocalSamplerTest, GetSampleForwardsFatalServerError) {
  auto table = MakeTable();

  Sampler::Options options;
  options.rate_limiter_timeout = absl::Milliseconds(10);
  Sampler sampler(table, options);

  std::vector<tensorflow::Tensor> sample;
  auto status = sampler.GetNextSample(&sample);
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  sampler.Close();
}

TEST(GrpcSamplerTest, GetNextSampleForwardsFatalServerError) {
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
    status = sampler.GetNextSample(&sample);
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

TEST(GrpcSamplerTest, GetNextTimestepRetriesTransientErrors) {
  const int kNumWorkers = 2;
  const int kItemLength = 10;
  const auto kError = grpc::Status(grpc::StatusCode::UNAVAILABLE, "");

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

TEST(GrpcSamplerTest, GetNextSampleRetriesTransientErrors) {
  const int kNumWorkers = 2;
  const int kItemLength = 10;
  const auto kError = grpc::Status(grpc::StatusCode::UNAVAILABLE, "");

  auto stub = MakeFlakyStub(
      {MakeResponse(kItemLength), MakeResponse(kItemLength)}, {kError});
  Sampler sampler(stub, "table",
                  {Sampler::kUnlimitedMaxSamples, 1, kNumWorkers});

  // It is possible that the sample returned by one of the workers is reached
  // before the failing worker has reported it's error so we need to pop at
  // least two samples to ensure that the we will see the error.
  for (int i = 0; i < 2; i++) {
    std::vector<tensorflow::Tensor> sample;
    REVERB_EXPECT_OK(sampler.GetNextSample(&sample));
  }
}

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

  // Add a column of length 10 to the existing one of length 5.
  CompressTensorAsProto(MakeTensor(10),
                        response.add_data()->mutable_data()->add_tensors());
  auto* slice = response.mutable_info()
                    ->mutable_item()
                    ->mutable_flat_trajectory()
                    ->add_columns()
                    ->add_chunk_slices();
  *slice = response.info().item().flat_trajectory().columns(0).chunk_slices(0);
  slice->set_index(1);
  slice->set_length(10);

  auto stub = MakeGoodStub({std::move(response)});
  Sampler sampler(stub, "table", {2, 1, 1});
  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;
  auto status = sampler.GetNextTimestep(&sample, &end_of_sequence);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
}

TEST(GrpcSamplerTest, GetNextSampleReturnsErrorIfMaximumSamplesExceeded) {
  auto stub = MakeGoodStub({MakeResponse(5), MakeResponse(5), MakeResponse(5)});
  Sampler sampler(stub, "table", {2, 1, 1});
  std::vector<tensorflow::Tensor> sample;
  REVERB_EXPECT_OK(sampler.GetNextSample(&sample));
  REVERB_EXPECT_OK(sampler.GetNextSample(&sample));
  EXPECT_EQ(sampler.GetNextSample(&sample).code(),
            absl::StatusCode::kOutOfRange);
}

TEST(LocalSamplerTest, GetNextSampleReturnsErrorIfMaximumSamplesExceeded) {
  auto table = MakeTable();
  InsertItem(table.get(), 1, 1.0, {2});
  InsertItem(table.get(), 2, 1.0, {2});
  InsertItem(table.get(), 3, 1.0, {2});

  Sampler sampler(table, {2});

  std::vector<tensorflow::Tensor> sample;
  REVERB_EXPECT_OK(sampler.GetNextSample(&sample));
  REVERB_EXPECT_OK(sampler.GetNextSample(&sample));
  EXPECT_EQ(sampler.GetNextSample(&sample).code(),
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

TEST(SamplerOptionsTest, ValidateChecksFlexibleBatchSize) {
  Sampler::Options options;
  options.flexible_batch_size = 0;
  EXPECT_EQ(options.Validate().code(), absl::StatusCode::kInvalidArgument);
  options.flexible_batch_size = Sampler::kAutoSelectValue;
  REVERB_EXPECT_OK(options.Validate());
  options.flexible_batch_size = -2;
  EXPECT_EQ(options.Validate().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
