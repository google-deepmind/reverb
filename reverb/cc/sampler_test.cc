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
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/reverb_service_mock.grpc.pb.h"
#include "reverb/cc/tensor_compression.h"
#include "reverb/cc/testing/tensor_testutil.h"
#include "reverb/cc/testing/time_testutil.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace deepmind {
namespace reverb {
namespace {

using test::ExpectTensorEqual;
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

SampleStreamResponse MakeResponse(int item_length, bool delta_encode = false,
                                  int offset = 0, int data_length = 0) {
  if (data_length == 0) {
    data_length = item_length;
  }
  REVERB_CHECK_LE(item_length + offset, data_length);

  SampleStreamResponse response;
  response.mutable_info()->mutable_item()->mutable_sequence_range()->set_length(
      item_length);
  response.mutable_info()->mutable_item()->mutable_sequence_range()->set_offset(
      offset);
  auto tensor = MakeTensor(data_length);
  if (delta_encode) {
    tensor = DeltaEncode(tensor, true);
    response.mutable_data()->set_delta_encoded(true);
  }

  CompressTensorAsProto(tensor, response.mutable_data()->add_data());
  return response;
}

TEST(SamplerTest, SendsFirstRequest) {
  auto stub = MakeGoodStub({MakeResponse(1)});
  Sampler sampler(stub, "table", {1, 1, 1});
  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;
  TF_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  EXPECT_THAT(stub->requests(), SizeIs(1));
}

TEST(SamplerTest, SetsEndOfSequence) {
  auto stub = MakeGoodStub({MakeResponse(2), MakeResponse(1)});
  Sampler sampler(stub, "table", {2, 1});

  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;

  // First sequence has 2 timesteps so first timestep should not be the end of
  // a sequence.
  TF_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  absl::SleepFor(absl::Milliseconds(5));
  EXPECT_FALSE(end_of_sequence);

  // Second timestep is the end of the first sequence.
  TF_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  absl::SleepFor(absl::Milliseconds(5));
  EXPECT_TRUE(end_of_sequence);

  // Third timestep is the first and only timestep of the second sequence.
  TF_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  absl::SleepFor(absl::Milliseconds(5));
  EXPECT_TRUE(end_of_sequence);
}

TEST(SamplerTest, GetNextSampleReturnsWholeSequence) {
  auto stub = MakeGoodStub({MakeResponse(5), MakeResponse(3)});
  Sampler sampler(stub, "table", {2, 1});

  std::vector<tensorflow::Tensor> first;
  TF_EXPECT_OK(sampler.GetNextSample(&first));
  EXPECT_THAT(first, SizeIs(4));  // ID, probability, table size, data.
  ExpectTensorEqual<tensorflow::uint64>(first[3], MakeTensor(5));

  std::vector<tensorflow::Tensor> second;
  TF_EXPECT_OK(sampler.GetNextSample(&second));
  EXPECT_THAT(second, SizeIs(4));  // ID, probability, table size, data.
  ExpectTensorEqual<tensorflow::uint64>(second[3], MakeTensor(3));
}

TEST(SamplerTest, GetNextSampleTrimsSequence) {
  auto stub = MakeGoodStub({
      MakeResponse(5, false, 1, 6),   // Trim offset at the start.
      MakeResponse(3, false, 0, 4),   // Trim timestep from end.
      MakeResponse(2, false, 1, 10),  // Trim offset and end.
  });
  Sampler sampler(stub, "table", {3, 1});

  std::vector<tensorflow::Tensor> start_trimmed;
  TF_EXPECT_OK(sampler.GetNextSample(&start_trimmed));
  ASSERT_THAT(start_trimmed, SizeIs(4));  // ID, probability, table size, data.
  ExpectTensorEqual<tensorflow::uint64>(
      start_trimmed[3],
      tensorflow::tensor::DeepCopy(MakeTensor(6).Slice(1, 6)));

  std::vector<tensorflow::Tensor> end_trimmed;
  TF_EXPECT_OK(sampler.GetNextSample(&end_trimmed));
  ASSERT_THAT(end_trimmed, SizeIs(4));  // ID, probability, table size, data.
  ExpectTensorEqual<tensorflow::uint64>(end_trimmed[3],
                                        MakeTensor(4).Slice(0, 3));

  std::vector<tensorflow::Tensor> start_and_end_trimmed;
  TF_EXPECT_OK(sampler.GetNextSample(&start_and_end_trimmed));
  ASSERT_THAT(start_and_end_trimmed,
              SizeIs(4));  // ID, probability, table size, data.
  ExpectTensorEqual<tensorflow::uint64>(
      start_and_end_trimmed[3],
      tensorflow::tensor::DeepCopy(MakeTensor(10).Slice(1, 3)));
}

TEST(SamplerTest, RespectsBufferSizeAndMaxSamples) {
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
    TF_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  }

  test::WaitFor([&]() { return stub->requests().size() == 1; },
                absl::Milliseconds(10), 100);
  EXPECT_THAT(stub->requests(), SizeIs(1));

  // The 10th sample (+1 in the queue) mean that all the requested samples
  // have been received and thus a new request is sent to retrieve more.
  TF_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
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
    TF_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  }
  test::WaitFor([&]() { return stub->requests().size() == 2; },
                absl::Milliseconds(10), 100);
  EXPECT_THAT(stub->requests(), SizeIs(2));
}

TEST(SamplerTest, UnpacksDeltaEncodedTensors) {
  auto stub = MakeGoodStub({MakeResponse(10, false), MakeResponse(10, true)});
  Sampler sampler(stub, "table", {2, 1});
  std::vector<tensorflow::Tensor> not_encoded;
  std::vector<tensorflow::Tensor> encoded;
  TF_EXPECT_OK(sampler.GetNextSample(&not_encoded));
  TF_EXPECT_OK(sampler.GetNextSample(&encoded));
  ASSERT_EQ(not_encoded.size(), encoded.size());
  EXPECT_EQ(encoded[0].dtype(), tensorflow::DT_UINT64);
  for (int i = 3; i < encoded.size(); i++) {
    ExpectTensorEqual<tensorflow::uint64>(encoded[i], not_encoded[i]);
  }
}

TEST(SamplerTest, GetNextTimestepForwardsFatalServerError) {
  const int kNumWorkers = 4;
  const int kItemLength = 10;
  const auto kError = grpc::Status(grpc::StatusCode::NOT_FOUND, "");

  auto stub = MakeFlakyStub({MakeResponse(kItemLength)}, {kError});
  Sampler sampler(stub, "table",
                  {Sampler::kUnlimitedMaxSamples, 1, kNumWorkers});

  // It is possible that the sample returned by one of the workers is reached
  // before the failing worker has reported it's error so we need to pop at
  // least two samples to ensure that the we will see the error.
  tensorflow::Status status;
  for (int i = 0; status.ok() && i < kItemLength + 1; i++) {
    std::vector<tensorflow::Tensor> sample;
    bool end_of_sequence;
    status = sampler.GetNextTimestep(&sample, &end_of_sequence);
  }
  EXPECT_EQ(status.code(), tensorflow::error::NOT_FOUND);
  sampler.Close();
}

TEST(SamplerTest, GetNextSampleForwardsFatalServerError) {
  const int kNumWorkers = 4;
  const int kItemLength = 10;
  const auto kError = grpc::Status(grpc::StatusCode::NOT_FOUND, "");

  auto stub = MakeFlakyStub({MakeResponse(kItemLength)}, {kError});
  Sampler sampler(stub, "table",
                  {Sampler::kUnlimitedMaxSamples, 1, kNumWorkers});

  // It is possible that the sample returned by one of the workers is reached
  // before the failing worker has reported it's error so we need to pop at
  // least two samples to ensure that the we will see the error.
  tensorflow::Status status;
  for (int i = 0; status.ok() && i < 2; i++) {
    std::vector<tensorflow::Tensor> sample;
    status = sampler.GetNextSample(&sample);
  }
  EXPECT_EQ(status.code(), tensorflow::error::NOT_FOUND);
}

TEST(SamplerTest, GetNextTimestepRetriesTransientErrors) {
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
    TF_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  }
}

TEST(SamplerTest, GetNextSampleRetriesTransientErrors) {
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
    TF_EXPECT_OK(sampler.GetNextSample(&sample));
  }
}

TEST(SamplerTest, GetNextTimestepReturnsErrorIfMaximumSamplesExceeded) {
  auto stub = MakeGoodStub({MakeResponse(1), MakeResponse(1), MakeResponse(1)});
  Sampler sampler(stub, "table", {2, 1, 1});
  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;
  TF_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  TF_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  EXPECT_EQ(sampler.GetNextTimestep(&sample, &end_of_sequence).code(),
            tensorflow::error::OUT_OF_RANGE);
}

TEST(SamplerTest, GetNextSampleReturnsErrorIfMaximumSamplesExceeded) {
  auto stub = MakeGoodStub({MakeResponse(5), MakeResponse(5), MakeResponse(5)});
  Sampler sampler(stub, "table", {2, 1, 1});
  std::vector<tensorflow::Tensor> sample;
  TF_EXPECT_OK(sampler.GetNextSample(&sample));
  TF_EXPECT_OK(sampler.GetNextSample(&sample));
  EXPECT_EQ(sampler.GetNextSample(&sample).code(),
            tensorflow::error::OUT_OF_RANGE);
}

TEST(SamplerTest, StressTestWithoutErrors) {
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
    TF_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  }

  // There should be no more samples left.
  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;
  EXPECT_EQ(sampler.GetNextTimestep(&sample, &end_of_sequence).code(),
            tensorflow::error::OUT_OF_RANGE);
}

TEST(SamplerTest, StressTestWithTransientErrors) {
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
    TF_EXPECT_OK(sampler.GetNextTimestep(&sample, &end_of_sequence));
  }

  // There should be no more samples left.
  std::vector<tensorflow::Tensor> sample;
  bool end_of_sequence;
  EXPECT_EQ(sampler.GetNextTimestep(&sample, &end_of_sequence).code(),
            tensorflow::error::OUT_OF_RANGE);
}

TEST(SamplerDeathTest, DiesIfMaxInFlightSamplesPerWorkerIsNonPositive) {
  Sampler::Options options;

  options.max_in_flight_samples_per_worker = 0;
  ASSERT_DEATH(Sampler sampler(MakeGoodStub({}), "table", options), "");

  options.max_in_flight_samples_per_worker = -1;
  ASSERT_DEATH(Sampler sampler(MakeGoodStub({}), "table", options), "");
}

TEST(SamplerDeathTest, DiesIfMaxSamplesInvalid) {
  Sampler::Options options;

  options.max_samples = -2;
  ASSERT_DEATH(Sampler sampler(MakeGoodStub({}), "table", options), "");

  options.max_samples = 0;
  ASSERT_DEATH(Sampler sampler(MakeGoodStub({}), "table", options), "");
}

TEST(SamplerDeathTest, DiesIfNumWorkersIsInvalid) {
  Sampler::Options options;

  options.num_workers = 0;
  ASSERT_DEATH(Sampler sampler(MakeGoodStub({}), "table", options), "");

  options.num_workers = -2;
  ASSERT_DEATH(Sampler sampler(MakeGoodStub({}), "table", options), "");
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
