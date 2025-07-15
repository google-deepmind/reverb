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

#include "reverb/cc/trajectory_writer.h"

#include <grpcpp/support/status.h>

#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include "grpcpp/test/mock_stream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/reverb_service_mock.grpc.pb.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/queue.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "reverb/cc/testing/tensor_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"

namespace deepmind {
namespace reverb {
namespace {

using ::testing::ElementsAre;
using ::testing::Return;
using ::testing::ReturnNew;
using ::testing::UnorderedElementsAre;

using Step = ::std::vector<::absl::optional<::tensorflow::Tensor>>;
using StepRef = ::std::vector<::absl::optional<::std::weak_ptr<CellRef>>>;

const auto kIntSpec = internal::TensorSpec{"0", tensorflow::DT_INT32, {1}};
const auto kFloatSpec = internal::TensorSpec{"0", tensorflow::DT_FLOAT, {1}};

MATCHER(IsChunk, "") { return arg.chunks_size() == 1; }

MATCHER(IsChunkAndItem, "") {
  return arg.chunks_size() == 1 && arg.items_size() > 0; }

MATCHER(IsChunkAndItemPtr, "") {
  return arg->chunks_size() == 1 && arg->items_size() > 0;
}

MATCHER_P2(HasNumChunksAndItems, chunks, items, "") {
  return arg.chunks_size() == chunks && arg.items_size() == items;
}

MATCHER(IsItem, "") { return arg.items_size() > 0; }

inline std::string Int32Str() {
  return tensorflow::DataTypeString(tensorflow::DT_INT32);
}

inline tensorflow::Tensor MakeTensor(const internal::TensorSpec& spec) {
  if (spec.shape.dims() < 1) {
    return tensorflow::Tensor(spec.dtype, {});
  }

  tensorflow::TensorShape shape;
  REVERB_CHECK(spec.shape.AsTensorShape(&shape));
  tensorflow::Tensor tensor(spec.dtype, shape);

  for (int i = 0; i < tensor.NumElements(); i++) {
    if (spec.dtype == tensorflow::DT_FLOAT) {
      tensor.flat<float>()(i) = i;
    } else if (spec.dtype == tensorflow::DT_INT32) {
      tensor.flat<int32_t>()(i) = i;
    } else if (spec.dtype == tensorflow::DT_DOUBLE) {
      tensor.flat<double>()(i) = i;
    } else {
      REVERB_LOG(REVERB_FATAL) << "Unexpeted dtype";
    }
  }

  return tensor;
}

inline tensorflow::Tensor MakeRandomTensor(const internal::TensorSpec& spec) {
  auto tensor = MakeTensor(spec);

  absl::BitGen bit_gen;
  for (int i = 0; i < tensor.NumElements(); i++) {
    if (spec.dtype == tensorflow::DT_INT32) {
      tensor.flat<int32_t>()(i) = absl::Uniform<int32_t>(
          bit_gen, 0, std::numeric_limits<int32_t>::max());
    } else {
      REVERB_LOG(REVERB_FATAL) << "Unexpeted dtype";
    }
  }

  return tensor;
}

std::vector<TrajectoryColumn> MakeTrajectory(
    std::vector<std::vector<absl::optional<std::weak_ptr<CellRef>>>>
        trajectory) {
  std::vector<TrajectoryColumn> columns;
  for (const auto& optional_refs : trajectory) {
    std::vector<std::weak_ptr<CellRef>> col_refs;
    for (const auto& optional_ref : optional_refs) {
      col_refs.push_back(optional_ref.value());
    }
    columns.push_back(TrajectoryColumn(std::move(col_refs), /*squeeze=*/false));
  }
  return columns;
}

class FakeStream : public ::grpc::ClientCallbackReaderWriter<
                       ::deepmind::reverb::InsertStreamRequest,
                       ::deepmind::reverb::InsertStreamResponse> {
 public:
  FakeStream(bool generate_responses, absl::Status status)
      : generate_responses_(generate_responses), status_(ToGrpcStatus(status)) {
    requests_ = std::make_shared<std::vector<InsertStreamRequest>>();
  }

  void SetReactor(
      ::grpc::ClientBidiReactor<::deepmind::reverb::InsertStreamRequest,
                                ::deepmind::reverb::InsertStreamResponse>*
          reactor) {
    reactor_ = reactor;
    BindReactor(reactor);
  }
  MOCK_METHOD(void, StartCall, ());
  MOCK_METHOD(void, WritesDone, ());
  MOCK_METHOD(void, AddHold, (int holds));
  void Read(::deepmind::reverb::InsertStreamResponse* resp) {
    absl::MutexLock lock(&mu_);
    REVERB_CHECK(response_ == nullptr);
    response_ = resp;
  }

  void RemoveHold() {
    callback_thread_ = internal::StartThread("OnDoneAsync", [this] {
      reactor_->OnDone(status_);
    });
  }

  void Write(const InsertStreamRequest* msg,
             grpc::WriteOptions options) override {
    int confirm_cnt = 0;
    {
      absl::MutexLock lock(&mu_);
      requests_->push_back(*msg);
      for (auto& item : msg->items()) {
        pending_confirmation_.push(item.key());
        confirm_cnt++;
      }
      if (!generate_responses_) {
        return;
      }
    }
    reactor_->OnWriteDone(true);
    ConfirmItems(confirm_cnt);
  }

  void SetStatus(grpc::Status status) {
    status_ = status;
  }

  void BlockUntilNumRequestsIs(int size) const {
    absl::MutexLock lock(&mu_);
    auto trigger = [size, this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return requests_->size() == size;
    };
    mu_.Await(absl::Condition(&trigger));
  }

  void ConfirmItems(int count) {
    {
      absl::MutexLock lock(&mu_);
      for (int x = 0; x < count; x++) {
        response_->add_keys(pending_confirmation_.front());
        pending_confirmation_.pop();
      }
      response_ = nullptr;
    }
    reactor_->OnReadDone(true);
  }

  const std::vector<InsertStreamRequest>& requests() const {
    absl::MutexLock lock(&mu_);
    return *requests_;
  }

  const int requests_size() const {
    absl::MutexLock lock(&mu_);
    return requests_->size();
  }

  InsertStreamRequest request(int idx) const {
    absl::MutexLock lock(&mu_);
    return (*requests_)[idx];
  }

  std::shared_ptr<std::vector<InsertStreamRequest>> requests_ptr() const {
    absl::MutexLock lock(&mu_);
    return requests_;
  }

 private:
  mutable absl::Mutex mu_;
  ::grpc::ClientBidiReactor<::deepmind::reverb::InsertStreamRequest,
                            ::deepmind::reverb::InsertStreamResponse>*
      reactor_ = nullptr;
  std::unique_ptr<::deepmind::reverb::internal::Thread> callback_thread_;
  std::shared_ptr<std::vector<InsertStreamRequest>> requests_
      ABSL_GUARDED_BY(mu_);
  std::queue<uint64_t> pending_confirmation_;
  ::deepmind::reverb::InsertStreamResponse* response_ ABSL_GUARDED_BY(mu_) =
      nullptr;
  const bool generate_responses_;
  grpc::Status status_;
};

class AsyncInterface : public ::deepmind::reverb::/* grpc_gen:: */ReverbService::
                           StubInterface::async_interface {
 public:
  AsyncInterface(bool generate_responses = true,
                 absl::Status status = absl::OkStatus())
      : stream_(generate_responses, status) {}

  void InsertStream(
      ::grpc::ClientContext* context,
      ::grpc::ClientBidiReactor<::deepmind::reverb::InsertStreamRequest,
                                ::deepmind::reverb::InsertStreamResponse>*
          reactor) {
    stream_.SetReactor(reactor);
  }

  MOCK_METHOD(void, Checkpoint,
               (::grpc::ClientContext*,
                    const ::deepmind::reverb::CheckpointRequest*,
                    ::deepmind::reverb::CheckpointResponse*,
                    std::function<void(::grpc::Status)>));
  MOCK_METHOD(void, Checkpoint,
               (::grpc::ClientContext*,
                    const ::deepmind::reverb::CheckpointRequest*,
                    ::deepmind::reverb::CheckpointResponse*,
                    ::grpc::ClientUnaryReactor* reactor));
  MOCK_METHOD(void, MutatePriorities,
               (::grpc::ClientContext*,
                    const ::deepmind::reverb::MutatePrioritiesRequest* request,
                    ::deepmind::reverb::MutatePrioritiesResponse* response,
                    std::function<void(::grpc::Status)>));
  MOCK_METHOD(void, MutatePriorities,
               (::grpc::ClientContext*,
                    const ::deepmind::reverb::MutatePrioritiesRequest* request,
                    ::deepmind::reverb::MutatePrioritiesResponse* response,
                    ::grpc::ClientUnaryReactor* reactor));
  MOCK_METHOD(void, Reset, (::grpc::ClientContext*,
                           const ::deepmind::reverb::ResetRequest* request,
                           ::deepmind::reverb::ResetResponse* response,
                           std::function<void(::grpc::Status)>));
  MOCK_METHOD(void, Reset, (::grpc::ClientContext*,
                           const ::deepmind::reverb::ResetRequest* request,
                           ::deepmind::reverb::ResetResponse* response,
                           ::grpc::ClientUnaryReactor* reactor));
  MOCK_METHOD(
      void, SampleStream,
      ((::grpc::ClientContext*),
       (::grpc::ClientBidiReactor<::deepmind::reverb::SampleStreamRequest,
                                  ::deepmind::reverb::SampleStreamResponse>*)));
  MOCK_METHOD(void, ServerInfo,
               (::grpc::ClientContext*,
                    const ::deepmind::reverb::ServerInfoRequest* request,
                    ::deepmind::reverb::ServerInfoResponse* response,
                    std::function<void(::grpc::Status)>));
  MOCK_METHOD(void, ServerInfo,
               (::grpc::ClientContext*,
                    const ::deepmind::reverb::ServerInfoRequest* request,
                    ::deepmind::reverb::ServerInfoResponse* response,
                    ::grpc::ClientUnaryReactor* reactor));
  MOCK_METHOD(void,
      InitializeConnection,
      ((::grpc::ClientContext*) ,
           (::grpc::ClientBidiReactor<
               ::deepmind::reverb::InitializeConnectionRequest,
               ::deepmind::reverb::InitializeConnectionResponse>*)));

 public:
  FakeStream stream_;
};

class MockReverbServiceAsyncStub
    : public ::deepmind::reverb::/* grpc_gen:: */ReverbService::StubInterface {
 public:
  MOCK_METHOD(::grpc::Status,
      Checkpoint,
      (::grpc::ClientContext*,
                     const ::deepmind::reverb::CheckpointRequest&,
                     ::deepmind::reverb::CheckpointResponse*));
  MOCK_METHOD(::grpc::ClientAsyncResponseReaderInterface<
                   ::deepmind::reverb::CheckpointResponse>*, AsyncCheckpointRaw,
               (
                   ::grpc::ClientContext*,
                   const ::deepmind::reverb::CheckpointRequest&,
                   ::grpc::CompletionQueue*));
  MOCK_METHOD(::grpc::ClientAsyncResponseReaderInterface<
                  ::deepmind::reverb::CheckpointResponse>*,
              PrepareAsyncCheckpointRaw,
              (::grpc::ClientContext*,
               const ::deepmind::reverb::CheckpointRequest&,
               ::grpc::CompletionQueue*));
  MOCK_METHOD((::grpc::ClientReaderWriterInterface<
                  ::deepmind::reverb::InsertStreamRequest,
                  ::deepmind::reverb::InsertStreamResponse>*),
              InsertStreamRaw, (::grpc::ClientContext*));
  MOCK_METHOD((::grpc::ClientAsyncReaderWriterInterface<
                  ::deepmind::reverb::InsertStreamRequest,
                  ::deepmind::reverb::InsertStreamResponse>*),
              AsyncInsertStreamRaw,
              (::grpc::ClientContext*, ::grpc::CompletionQueue*, void* tag));
  MOCK_METHOD((::grpc::ClientAsyncReaderWriterInterface<
                  ::deepmind::reverb::InsertStreamRequest,
                  ::deepmind::reverb::InsertStreamResponse>*),
              PrepareAsyncInsertStreamRaw,
              (::grpc::ClientContext*, ::grpc::CompletionQueue*));
  MOCK_METHOD(::grpc::Status,
      MutatePriorities,
      (::grpc::ClientContext*,
                     const ::deepmind::reverb::MutatePrioritiesRequest& request,
                     ::deepmind::reverb::MutatePrioritiesResponse* response));
  MOCK_METHOD(::grpc::ClientAsyncResponseReaderInterface<
                  ::deepmind::reverb::MutatePrioritiesResponse>*,
              AsyncMutatePrioritiesRaw,
              (::grpc::ClientContext*,
               const ::deepmind::reverb::MutatePrioritiesRequest& request,
               ::grpc::CompletionQueue*));
  MOCK_METHOD(::grpc::ClientAsyncResponseReaderInterface<
                  ::deepmind::reverb::MutatePrioritiesResponse>*,
              PrepareAsyncMutatePrioritiesRaw,
              (::grpc::ClientContext*,
               const ::deepmind::reverb::MutatePrioritiesRequest& request,
               ::grpc::CompletionQueue*));
  MOCK_METHOD(::grpc::Status, Reset,
              (::grpc::ClientContext*,
               const ::deepmind::reverb::ResetRequest& request,
               ::deepmind::reverb::ResetResponse* response));
  MOCK_METHOD(::grpc::ClientAsyncResponseReaderInterface<
                  ::deepmind::reverb::ResetResponse>*,
              AsyncResetRaw,
              (::grpc::ClientContext*,
               const ::deepmind::reverb::ResetRequest& request,
               ::grpc::CompletionQueue*));
  MOCK_METHOD((::grpc::ClientAsyncResponseReaderInterface<
                  ::deepmind::reverb::ResetResponse>*),
              PrepareAsyncResetRaw,
              (::grpc::ClientContext*,
               const ::deepmind::reverb::ResetRequest& request,
               ::grpc::CompletionQueue*));
  MOCK_METHOD((::grpc::ClientReaderWriterInterface<
                  ::deepmind::reverb::SampleStreamRequest,
                  ::deepmind::reverb::SampleStreamResponse>*),
              SampleStreamRaw, (::grpc::ClientContext*));
  MOCK_METHOD((::grpc::ClientAsyncReaderWriterInterface<
                  ::deepmind::reverb::SampleStreamRequest,
                  ::deepmind::reverb::SampleStreamResponse>*),
              AsyncSampleStreamRaw,
              (::grpc::ClientContext*, ::grpc::CompletionQueue*, void* tag));
  MOCK_METHOD((::grpc::ClientAsyncReaderWriterInterface<
                  ::deepmind::reverb::SampleStreamRequest,
                  ::deepmind::reverb::SampleStreamResponse>*),
              PrepareAsyncSampleStreamRaw,
              (::grpc::ClientContext*, ::grpc::CompletionQueue*));
  MOCK_METHOD(::grpc::Status, ServerInfo,
              (::grpc::ClientContext*,
               const ::deepmind::reverb::ServerInfoRequest&,
               ::deepmind::reverb::ServerInfoResponse* response));
  MOCK_METHOD(::grpc::ClientAsyncResponseReaderInterface<
                  ::deepmind::reverb::ServerInfoResponse>*,
              AsyncServerInfoRaw,
              (::grpc::ClientContext*,
               const ::deepmind::reverb::ServerInfoRequest&,
               ::grpc::CompletionQueue*));
  MOCK_METHOD(::grpc::ClientAsyncResponseReaderInterface<
                  ::deepmind::reverb::ServerInfoResponse>*,
              PrepareAsyncServerInfoRaw,
              (::grpc::ClientContext*,
               const ::deepmind::reverb::ServerInfoRequest&,
               ::grpc::CompletionQueue*));
  MOCK_METHOD((::grpc::ClientReaderWriterInterface<
                  ::deepmind::reverb::InitializeConnectionRequest,
                  ::deepmind::reverb::InitializeConnectionResponse>*),
              InitializeConnectionRaw, (::grpc::ClientContext*));
  MOCK_METHOD((::grpc::ClientAsyncReaderWriterInterface<
                  ::deepmind::reverb::InitializeConnectionRequest,
                  ::deepmind::reverb::InitializeConnectionResponse>*),
              AsyncInitializeConnectionRaw,
              (::grpc::ClientContext*, ::grpc::CompletionQueue*, void*));
  MOCK_METHOD((::grpc::ClientAsyncReaderWriterInterface<
                  ::deepmind::reverb::InitializeConnectionRequest,
                  ::deepmind::reverb::InitializeConnectionResponse>*),
              PrepareAsyncInitializeConnectionRaw,
              (::grpc::ClientContext*, ::grpc::CompletionQueue*));
  MOCK_METHOD(deepmind::reverb::/* grpc_gen:: */ReverbService::StubInterface::
                  async_interface*,
              async, ());
};

inline TrajectoryWriter::Options MakeOptions(int max_chunk_length,
                                             int num_keep_alive_refs) {
  return TrajectoryWriter::Options{
      .chunker_options = std::make_shared<ConstantChunkerOptions>(
          max_chunk_length, num_keep_alive_refs),
  };
}

TEST(TrajectoryWriter, AppendValidatesDtype) {
  AsyncInterface async;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillRepeatedly(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/10, /*num_keep_alive_refs=*/10));
  StepRef refs;

  // Initiate the spec with the first step.
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kFloatSpec)}), &refs));

  // Change the dtypes in the next step.
  auto status =
      writer.Append(Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}), &refs);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  absl::StrCat("Tensor of wrong dtype provided for column 1. "
                               "Got ",
                               Int32Str(), " but expected float.")));
}

TEST(TrajectoryWriter, AppendValidatesShapes) {
  AsyncInterface async;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillRepeatedly(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/10, /*num_keep_alive_refs=*/10));
  StepRef refs;

  // Initiate the spec with the first step.
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &refs));

  // Change the dtypes in the next step.
  auto status = writer.Append(Step({MakeTensor(internal::TensorSpec{
                                  kIntSpec.name, kIntSpec.dtype, {3}})}),
                              &refs);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Tensor of incompatible shape provided for column 0. "
                  "Got [3] which is incompatible with [1]."));
}

TEST(TrajectoryWriter, AppendAcceptsPartialSteps) {
  AsyncInterface async;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillRepeatedly(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/10, /*num_keep_alive_refs=*/10));

  // Initiate the spec with the first step.
  StepRef both;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kFloatSpec)}), &both));

  // Only append to the first column.
  StepRef first_column_only;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec), absl::nullopt}),
                                 &first_column_only));
  EXPECT_FALSE(first_column_only[1].has_value());
}

TEST(TrajectoryWriter, AppendPartialRejectsMultipleUsesOfSameColumn) {
  AsyncInterface async;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillRepeatedly(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/10, /*num_keep_alive_refs=*/10));

  // Append first column only.
  StepRef first_column_only;
  REVERB_ASSERT_OK(
      writer.AppendPartial(Step({MakeTensor(kIntSpec)}), &first_column_only));

  // Appending the second column only should be fine.
  StepRef second_column_only;
  REVERB_ASSERT_OK(writer.AppendPartial(
      Step({absl::nullopt, MakeTensor(kFloatSpec)}), &second_column_only));

  // Appending the first column again should not be allowed.
  StepRef first_column_again;
  auto status =
      writer.AppendPartial(Step({MakeTensor(kIntSpec)}), &first_column_again);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Append/AppendPartial called with data containing column "
                  "that was present in previous AppendPartial call."));
}

TEST(TrajectoryWriter, AppendRejectsColumnsProvidedInPreviousPartialCall) {
  AsyncInterface async;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillRepeatedly(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/10, /*num_keep_alive_refs=*/10));

  // Append first column only.
  StepRef first_column_only;
  REVERB_ASSERT_OK(
      writer.AppendPartial(Step({MakeTensor(kIntSpec)}), &first_column_only));

  // Finalize the step with BOTH columns. That is both the missing column and
  // the one that was already provided.
  StepRef both_columns;
  auto status = writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kFloatSpec)}), &both_columns);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Append/AppendPartial called with data containing column "
                  "that was present in previous AppendPartial call."));
}

TEST(TrajectoryWriter, AppendPartialDoesNotIncrementEpisodeStep) {
  AsyncInterface async;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillRepeatedly(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/10, /*num_keep_alive_refs=*/10));

  // Append first column only and keep the step open.
  StepRef first_column_only;
  REVERB_ASSERT_OK(
      writer.AppendPartial(Step({MakeTensor(kIntSpec)}), &first_column_only));

  // Append to the second column only and close the step.
  StepRef second_column_only;
  REVERB_ASSERT_OK(writer.Append(Step({absl::nullopt, MakeTensor(kFloatSpec)}),
                                 &second_column_only));

  // Since the step was kept open after the first call the second call should
  // result in the same episode step.
  EXPECT_EQ(first_column_only[0]->lock()->episode_step(),
            second_column_only[1]->lock()->episode_step());

  // If we do another call then the episode step should have changed since the
  // previous step was closed with the second call.
  StepRef first_column_only_again;
  REVERB_ASSERT_OK(
      writer.Append(Step({MakeTensor(kIntSpec)}), &first_column_only_again));
  EXPECT_EQ(first_column_only[0]->lock()->episode_step() + 1,
            first_column_only_again[0]->lock()->episode_step());
}

TEST(TrajectoryWriter, ConfigureChunkerOnExistingColumn) {
  AsyncInterface async;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillRepeatedly(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1));

  // Create the column with the first step.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));

  // The chunk should be created automatically since max_chunk_length is 1.
  EXPECT_TRUE(first[0]->lock()->IsReady());

  // Reconfigure the column to have a chunk length of 2 instead.
  REVERB_ASSERT_OK(writer.ConfigureChunker(
      0, std::make_shared<ConstantChunkerOptions>(/*max_chunk_length=*/2,
                                                  /*num_keep_alive_refs=*/2)));

  // Appending a second step should now NOT result in a being created.
  StepRef second;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &second));
  EXPECT_FALSE(second[0]->lock()->IsReady());

  // A third step should however result in the chunk being created. Also note
  // that two steps are alive instead of the orignially configured 1.
  StepRef third;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &third));
  EXPECT_TRUE(second[0]->lock()->IsReady());
  EXPECT_TRUE(third[0]->lock()->IsReady());
}

TEST(TrajectoryWriter, ConfigureChunkerOnFutureColumn) {
  AsyncInterface async;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillRepeatedly(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1));

  // Create the first column with the first step.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));

  // The chunk should be created automatically since max_chunk_length is 1.
  EXPECT_TRUE(first[0]->lock()->IsReady());

  // Configure the second column (not yet seen) to have max_chunk_length 2
  // instead of 1.
  REVERB_ASSERT_OK(writer.ConfigureChunker(
      1, std::make_shared<ConstantChunkerOptions>(/*max_chunk_length=*/2,
                                                  /*num_keep_alive_refs=*/2)));

  // Appending a second step should finalize the first column since it still has
  // max_chunk_length 1. The second column should however NOT be finalized since
  // it has max_chunk_length 2.
  StepRef second;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}), &second));
  EXPECT_TRUE(second[0]->lock()->IsReady());
  EXPECT_FALSE(second[1]->lock()->IsReady());

  // The first step should have expired now as well since num_keep_alive_refs is
  // 1 for the first column.
  EXPECT_TRUE(first[0]->expired());

  // When appending the third step we expect both columns to be finalized. We
  // also expect the first column in the second step to expire since its
  // num_keep_alive_refs is 1.
  StepRef third;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}), &third));
  EXPECT_TRUE(third[0]->lock()->IsReady());
  EXPECT_TRUE(third[1]->lock()->IsReady());
  EXPECT_TRUE(second[0]->expired());
  EXPECT_FALSE(second[1]->expired());
}

TEST(TrajectoryWriter, NoDataIsSentIfNoItemsCreated) {
  AsyncInterface async;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async()).WillRepeatedly(Return(&async));
  {
    TrajectoryWriter writer(
        stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1));
    StepRef refs;

    for (int i = 0; i < 10; ++i) {
      REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &refs));
    }
  }
  EXPECT_TRUE(async.stream_.requests().empty());
}

TEST(TrajectoryWriter, ItemSentStraightAwayIfChunksReady) {
  AsyncInterface async;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async()).WillOnce(Return(&async));

  TrajectoryWriter writer(stub, MakeOptions(/*max_chunk_length=*/1,
                                            /*num_keep_alive_refs=*/1));
  StepRef refs;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &refs));

  // Nothing sent before the item created.
  EXPECT_THAT(async.stream_.requests(), ::testing::IsEmpty());

  // The chunk is completed so inserting an item should result in both chunk
  // and item being sent.
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{refs[0]}})));

  async.stream_.BlockUntilNumRequestsIs(1);

  // Chunk is sent before item.
  EXPECT_THAT(async.stream_.requests(), ElementsAre(IsChunkAndItem()));

  // Adding a second item should result in the item being sent straight away.
  // Note that the chunk is not sent again.
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 0.5, MakeTrajectory({{refs[0]}})));

  async.stream_.BlockUntilNumRequestsIs(2);

  EXPECT_THAT(async.stream_.requests()[1], IsItem());
}

TEST(TrajectoryWriter, ItemIsSentWhenAllChunksDone) {
  AsyncInterface async;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async()).WillOnce(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2));

  // Write to both columns in the first step.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}), &first));

  // Create an item which references the first row in the two columns.
  REVERB_ASSERT_OK(writer.CreateItem("table", 1.0,
                                     MakeTrajectory({{first[0]}, {first[1]}})));

  // No data is sent yet since the chunks are not completed.
  EXPECT_THAT(async.stream_.requests(), ::testing::IsEmpty());

  // In the second step we only write to the first column. Only a single chunk
  // is being transmitted.
  StepRef second;
  REVERB_ASSERT_OK(
      writer.Append(Step({MakeTensor(kIntSpec), absl::nullopt}), &second));

  async.stream_.BlockUntilNumRequestsIs(1);
  EXPECT_THAT(async.stream_.requests(), ElementsAre(IsChunk()));

  // Writing to the first column again, even if we do it twice and trigger a new
  // chunk to be completed, should not trigger any new messages.
  for (int i = 0; i < 2; i++) {
    StepRef refs;
    REVERB_ASSERT_OK(
        writer.Append(Step({MakeTensor(kIntSpec), absl::nullopt}), &refs));
  }
  EXPECT_THAT(async.stream_.requests(), ElementsAre(IsChunk()));

  // Writing to the second column will trigger the completion of the chunk in
  // the second column. This in turn should trigger the transmission of the
  // second chunk and the item.
  StepRef third;
  REVERB_ASSERT_OK(
      writer.Append(Step({absl::nullopt, MakeTensor(kIntSpec)}), &third));

  async.stream_.BlockUntilNumRequestsIs(2);

  EXPECT_THAT(async.stream_.requests(),
              ElementsAre(IsChunk(), HasNumChunksAndItems(1, 1)));
}

TEST(TrajectoryWriter, ChunkersNotifiedWhenAllChunksDone) {
  class FakeChunkerOptions : public ChunkerOptions {
   public:
    FakeChunkerOptions(absl::BlockingCounter* counter) : counter_(counter) {}

    int GetMaxChunkLength() const override { return 1; }
    int GetNumKeepAliveRefs() const override { return 1; }
    bool GetDeltaEncode() const override { return false; }
    bool GetCompressionDisabled() const override { return false; }

    absl::Status OnItemFinalized(
        const PrioritizedItem& item,
        absl::Span<const std::shared_ptr<CellRef>> refs) override {
      counter_->DecrementCount();
      return absl::OkStatus();
    }

    std::shared_ptr<ChunkerOptions> Clone() const override {
      return std::make_shared<FakeChunkerOptions>(counter_);
    }

   private:
    absl::BlockingCounter* counter_;
  };

  AsyncInterface async;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async()).WillOnce(Return(&async));

  absl::BlockingCounter counter(2);
  TrajectoryWriter writer(stub,
                          {std::make_shared<FakeChunkerOptions>(&counter)});

  // Write to both columns in the first step.
  StepRef step;
  REVERB_ASSERT_OK(
      writer.Append(Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}), &step));

  // Create an item which references the step row in the two columns.
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{step[0]}, {step[1]}})));

  // The options should be cloned into the chunkers of the two columns and since
  // the chunk length is set to 1 the item should be finalized straight away and
  // the options notified.
  counter.Wait();
}

TEST(TrajectoryWriter, FlushSendsPendingItems) {
  AsyncInterface async;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async()).WillOnce(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2));

  // Write to both columns in the first step.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}), &first));

  // Create an item which references the first row in second column.
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[1]}})));

  // No data is sent yet since the chunks are not completed.
  EXPECT_THAT(async.stream_.requests(), ::testing::IsEmpty());

  // Calling flush should trigger the chunk creation of the second column only.
  // Since the first column isn't referenced by the pending item there is no
  // need for it to be prematurely finalized. Since all chunks required by the
  // pending item is now ready, the chunk and the item should be sent to the
  // server.
  REVERB_ASSERT_OK(writer.Flush());
  EXPECT_FALSE(first[0].value().lock()->IsReady());
  EXPECT_TRUE(first[1].value().lock()->IsReady());
  EXPECT_THAT(async.stream_.requests(), ElementsAre(IsChunkAndItem()));
}

TEST(TrajectoryWriter, DestructorFlushesPendingItems) {
  AsyncInterface async;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async()).WillOnce(Return(&async));

  // The requests vector needs to outlive the stream.
  auto requests = async.stream_.requests_ptr();
  {
    TrajectoryWriter writer(
        stub, MakeOptions(/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2));

    // Write to both columns in the first step.
    StepRef first;
    REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));

    // Create an item which references the first row in the incomplete chunk..
    REVERB_ASSERT_OK(
        writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}})));

    // No data is sent yet since the chunks are not completed.
    EXPECT_THAT(async.stream_.requests(), ::testing::IsEmpty());
  }

  EXPECT_THAT(*requests, ElementsAre(IsChunkAndItem()));
}

class ParameterizedTrajectoryWriterTest : public ::testing::Test,
                public ::testing::WithParamInterface<absl::Status> {};

TEST_P(ParameterizedTrajectoryWriterTest, RetriesOnTransientError) {
  AsyncInterface fail_stream(false, GetParam());
  AsyncInterface success_stream;

  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillOnce(Return(&fail_stream))
      .WillOnce(Return(&success_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1));

  // Create an item and wait for it to be confirmed.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}})));
  fail_stream.stream_.BlockUntilNumRequestsIs(1);
  writer.OnWriteDone(false);
  REVERB_ASSERT_OK(writer.Flush());

  // The first stream will fail on the first request (item). The writer should
  // then close the stream and once it sees the UNAVAILABLE error open a new
  // stream. The writer should then proceed to resend the chunk since there is
  // no guarantee that the new stream is connected to the same server and thus
  // the data might not exist on the server.
  success_stream.stream_.BlockUntilNumRequestsIs(1);
  EXPECT_THAT(success_stream.stream_.requests(), ElementsAre(IsChunkAndItem()));
}

TEST_P(ParameterizedTrajectoryWriterTest,
       RetriesNonConfirmedItemsOnTransientError) {
  AsyncInterface fail_stream(false, GetParam());
  AsyncInterface success_stream;

  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillOnce(Return(&fail_stream))
      .WillOnce(Return(&success_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1));

  // Create two items and wait for them to be confirmed.
  for (int i = 0; i < 2; i++) {
    StepRef step;
    REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));
    REVERB_ASSERT_OK(
        writer.CreateItem("table", 1.0, MakeTrajectory({{step[0]}})));
    fail_stream.stream_.BlockUntilNumRequestsIs(i + 1);
    writer.OnWriteDone(i == 0);
  }
  REVERB_ASSERT_OK(writer.Flush());

  // The first stream will fail on the second (item). The writer should
  // then close the stream and once it sees the UNAVAILABLE error open a new
  // stream. The writer should realise that the first item, even though
  // successfully written to the stream, never got confirmed by the server and
  // send it again when opening a new stream.
  success_stream.stream_.BlockUntilNumRequestsIs(1);
  EXPECT_THAT(success_stream.stream_.requests(),
              ElementsAre(HasNumChunksAndItems(2, 2)));
}

TEST_P(ParameterizedTrajectoryWriterTest,
     RetriesNonConfirmedItemsIfReaderStoppedAndErrorIsTransient) {
  AsyncInterface fail_stream(false, GetParam());
  AsyncInterface success_stream;

  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillOnce(Return(&fail_stream))
      .WillOnce(Return(&success_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1));

  // Create one item and wait for it to be confirmed.
  StepRef step;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{step[0]}})));
  fail_stream.stream_.BlockUntilNumRequestsIs(1);
  writer.OnReadDone(false);
  REVERB_ASSERT_OK(writer.Flush());

  // The first stream will accept the write but then fail when the reader worker
  // is trying to read the confirmation. We do this to simulate a setup were one
  // or more items were written to the server but the response is delayed due
  // the rate limiter (or similar blockage) and while waiting for the response
  // the server becomes unavailable. It is important that we are able to recover
  // from this in the same way as we recover from errors that are detected when
  // writing to the stream.
  //
  // The second stream will created and succeed.
  EXPECT_THAT(success_stream.stream_.requests(), ElementsAre(IsChunkAndItem()));
}

TEST_P(ParameterizedTrajectoryWriterTest, StopsOnNonTransientError) {
  AsyncInterface fail_stream(false, GetParam());

  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillOnce(Return(&fail_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1));

  // Create an item.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}})));
  fail_stream.stream_.BlockUntilNumRequestsIs(1);
  fail_stream.stream_.SetStatus(
      grpc::Status(grpc::StatusCode::INTERNAL, "A reason"));
  writer.OnReadDone(false);
  // Flushing should return the error encountered by the stream worker.
  auto flush_status = writer.Flush();
  EXPECT_EQ(flush_status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(std::string(flush_status.message()),
              ::testing::HasSubstr("A reason"));

  // The same error should be encountered in all methods.
  auto insert_status =
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}}));
  EXPECT_EQ(insert_status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(std::string(insert_status.message()),
              ::testing::HasSubstr("A reason"));

  auto append_status = writer.Append(Step({MakeTensor(kIntSpec)}), &first);
  EXPECT_EQ(append_status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(std::string(append_status.message()),
              ::testing::HasSubstr("A reason"));
}

TEST_P(ParameterizedTrajectoryWriterTest, FlushReturnsIfTimeoutExpired) {
  AsyncInterface fail_stream(false, GetParam());

  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillRepeatedly(Return(&fail_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1));

  // Create an item.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}})));

  // Flushing should return the error encountered by the stream worker.
  auto status =
      writer.Flush(/*ignore_last_num_items=*/0, absl::Milliseconds(100));
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr("Timeout exceeded"));

  // Close the writer to avoid having to mock the item confirmation response.
  writer.Close();
}

INSTANTIATE_TEST_CASE_P(ErrorTests, ParameterizedTrajectoryWriterTest,
                        ::testing::Values(absl::OkStatus(),
                                          absl::CancelledError(""),
                                          absl::UnavailableError("")));

TEST(TrajectoryWriter, FlushCanIgnorePendingItems) {
  AsyncInterface success_stream;

  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillOnce(Return(&success_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2));

  // Take a step with two columns.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}), &first));

  // Create two items, each referencing a separate column
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}})));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[1]}})));

  // Flushing should trigger the first item to be finalized and sent. The second
  // item should still be pending as its chunk have not yet been finalized.
  REVERB_ASSERT_OK(writer.Flush(/*ignore_last_num_items=*/1));

  // Only one item sent.
  EXPECT_THAT(success_stream.stream_.requests(), ElementsAre(IsChunkAndItem()));

  // The chunk of the first item is finalized while the other is not.
  EXPECT_TRUE(first[0]->lock()->IsReady());
  EXPECT_FALSE(first[1]->lock()->IsReady());
}

TEST(TrajectoryWriter, MultipleChunksAreSentInSameMessage) {
  AsyncInterface success_stream;

  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillOnce(Return(&success_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1));

  // Take a step with two columns.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}), &first));

  // Create an item referencing both columns. This will trigger the chunks for
  // both columns to be sent.
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[0], first[1]}})));
  REVERB_ASSERT_OK(writer.Flush());

  // Check that both chunks were sent in the same message.
  EXPECT_THAT(success_stream.stream_.requests(),
              ElementsAre(HasNumChunksAndItems(2, 1)));
}

TEST(TrajectoryWriter, MultipleRequestsSentWhenChunksLarge) {
  AsyncInterface success_stream;

  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillOnce(Return(&success_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1));

  // Take a step with three columns with really large random tensors. These do
  // not compress well which means that the ChunkData will be really large.
  internal::TensorSpec spec = {"0", tensorflow::DT_INT32, {8, 1024, 1024}};
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({
                                     MakeRandomTensor(spec),
                                     MakeRandomTensor(spec),
                                     MakeRandomTensor(spec),
                                 }),
                                 &first));

  // Create an item referencing both columns. This will trigger the chunks for
  // both columns to be sent.
  REVERB_ASSERT_OK(writer.CreateItem(
      "table", 1.0, MakeTrajectory({{first[0], first[1], first[2]}})));
  REVERB_ASSERT_OK(writer.Flush());

  // Each `ChunkData` should be ~32MB so the first two chunks should be grouped
  // together into a single message and the last one should be sent on its own.
  success_stream.stream_.BlockUntilNumRequestsIs(2);

  EXPECT_THAT(
      success_stream.stream_.requests(),
      ElementsAre(HasNumChunksAndItems(2, 0), HasNumChunksAndItems(1, 1)));
}

TEST(TrajectoryWriter, CreateItemRejectsExpiredCellRefs) {
  AsyncInterface success_stream;

  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillRepeatedly(Return(&success_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1));

  // Take two steps.
  StepRef first;
  StepRef second;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &second));

  // The num_keep_alive_refs is set to 1 so the first step has expired.
  auto status = writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}}));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Error in column 0: Column contains expired CellRef."));
}

TEST(TrajectoryWriter, KeepKeysOnlyIncludesStreamedKeys) {
  AsyncInterface success_stream;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillOnce(Return(&success_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1));

  // Create a step with two columns.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}), &first));

  // Create an item which only references one of the columns.
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}})));
  REVERB_ASSERT_OK(writer.Flush());

  // Only the chunk of the first column has been used (and thus streamed). The
  // server should thus only be instructed to keep the one chunk around.
  EXPECT_THAT(success_stream.stream_.requests(),
              UnorderedElementsAre(IsChunkAndItem()));
  EXPECT_THAT(success_stream.stream_.requests()[0].keep_chunk_keys(),
              UnorderedElementsAre(first[0].value().lock()->chunk_key()));
}

TEST(TrajectoryWriter, KeepKeysOnlyIncludesLiveChunks) {
  AsyncInterface success_stream;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillOnce(Return(&success_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/2));

  // Take a step and insert a trajectory.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}})));
  REVERB_ASSERT_OK(writer.Flush());

  // The one chunk that has been sent should be kept alive.
  EXPECT_THAT(success_stream.stream_.requests().back().keep_chunk_keys(),
              UnorderedElementsAre(first[0].value().lock()->chunk_key()));

  // Take a second step and insert a trajectory.
  StepRef second;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &second));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{second[0]}})));
  REVERB_ASSERT_OK(writer.Flush());

  // Both chunks should be kept alive since num_keep_alive_refs is 2.
  EXPECT_THAT(success_stream.stream_.requests().back().keep_chunk_keys(),
              UnorderedElementsAre(first[0].value().lock()->chunk_key(),
                                   second[0].value().lock()->chunk_key()));

  // Take a third step and insert a trajectory.
  StepRef third;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &third));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{third[0]}})));
  REVERB_ASSERT_OK(writer.Flush());

  // The chunk of the first step has now expired and thus the server no longer
  // need to keep it alive.
  EXPECT_THAT(success_stream.stream_.requests().back().keep_chunk_keys(),
              UnorderedElementsAre(second[0].value().lock()->chunk_key(),
                                   third[0].value().lock()->chunk_key()));
}

TEST(TrajectoryWriter, CreateItemValidatesTrajectoryDtype) {
  AsyncInterface success_stream;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillRepeatedly(Return(&success_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/2));

  // Take a step with two columns with different dtypes.
  StepRef step;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kFloatSpec)}), &step));

  // Create a trajectory where the two dtypes are used in the same column.
  auto status =
      writer.CreateItem("table", 1.0, MakeTrajectory({{step[0], step[1]}}));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  absl::StrCat("Error in column 0: Column references tensors "
                               "with different dtypes: ",
                               Int32Str(), " (index 0) != float (index 1).")));
}

TEST(TrajectoryWriter, CreateItemValidatesTrajectoryShapes) {
  AsyncInterface success_stream;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillRepeatedly(Return(&success_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/2));

  // Take a step with two columns with different shapes.
  StepRef step;

  REVERB_ASSERT_OK(writer.Append(
      Step({
          MakeTensor(kIntSpec),
          MakeTensor(internal::TensorSpec{"1", kIntSpec.dtype, {2}}),
      }),
      &step));

  // Create a trajectory where the two shapes are used in the same column.
  auto status =
      writer.CreateItem("table", 1.0, MakeTrajectory({{step[0], step[1]}}));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr("Error in column 0: Column references tensors with "
                           "incompatible shapes: [1] "
                           "(index 0) not compatible with [2] (index 1)."));
}

TEST(TrajectoryWriter, CreateItemValidatesTrajectoryNotEmpty) {
  AsyncInterface success_stream;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillRepeatedly(Return(&success_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1));

  StepRef step;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));

  // Create a trajectory without any columns.
  auto no_columns_status = writer.CreateItem("table", 1.0, {});
  EXPECT_EQ(no_columns_status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(no_columns_status.message()),
              ::testing::HasSubstr("trajectory must not be empty."));

  // Create a trajectory where all columns are empty.
  auto all_columns_empty_status = writer.CreateItem("table", 1.0, {{}, {}});
  EXPECT_EQ(all_columns_empty_status.code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(all_columns_empty_status.message()),
              ::testing::HasSubstr("trajectory must not be empty."));
}

TEST(TrajectoryWriter, CreateItemValidatesSqueezedColumns) {
  AsyncInterface success_stream;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
      .WillRepeatedly(Return(&success_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1));

  StepRef step;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));

  // Create a trajectory with a column that has two rows and is squeezed.
  auto status = writer.CreateItem(
      "table", 1.0,
      {TrajectoryColumn({step[0].value(), step[0].value()}, true)});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr("Error in column 0: TrajectoryColumn must contain "
                           "exactly one row when squeeze is set but got 2."));
}

TEST(TrajectoryWriter, CreateItemValidatesPriorityIsNotNan) {
  AsyncInterface success_stream;
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async()).WillRepeatedly(Return(&success_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1));

  StepRef step;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));

  auto status =
      writer.CreateItem("table", std::nan("1"), MakeTrajectory({step}));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("`priority` must not be nan."));
}

class TrajectoryWriterSignatureValidationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto stub = std::make_shared<MockReverbServiceAsyncStub>();
    EXPECT_CALL(*stub, async())
       .WillRepeatedly(Return(&success_stream_));

    TrajectoryWriter::Options options = {
        .chunker_options = std::make_shared<ConstantChunkerOptions>(1, 1),
        .flat_signature_map = internal::FlatSignatureMap({
            {
                "table",
                std::vector<internal::TensorSpec>({
                    internal::TensorSpec{
                        "first_col", tensorflow::DT_INT32, {2}},
                    internal::TensorSpec{
                        "second_col", tensorflow::DT_FLOAT, {1}},
                    internal::TensorSpec{
                        "var_length_col", tensorflow::DT_FLOAT, {-1}},
                }),
            },
        }),
    };
    writer_ = std::make_unique<TrajectoryWriter>(stub, options);

    // Take a step with enough columns to be able to compose both valid and
    // invalid trajectories.
    REVERB_ASSERT_OK(
        writer_->Append(Step({
                            MakeTensor({"0", tensorflow::DT_INT32, {}}),
                            MakeTensor({"1", tensorflow::DT_FLOAT, {}}),
                            MakeTensor({"2", tensorflow::DT_DOUBLE, {}}),
                            MakeTensor({"3", tensorflow::DT_FLOAT, {2, 2}}),
                        }),
                        &step_));
  }

  void TearDown() override {
    writer_->Close();
    writer_ = nullptr;
    step_.clear();
  }

  std::unique_ptr<TrajectoryWriter> writer_;
  StepRef step_;
  AsyncInterface success_stream_;
};

TEST_F(TrajectoryWriterSignatureValidationTest, Valid) {
  REVERB_EXPECT_OK(writer_->CreateItem("table", 1.0,
                                       MakeTrajectory({
                                           {step_[0], step_[0]},
                                           {step_[1]},
                                           {step_[1]},
                                       })));

  // Third column length is undefined so two rows should be just as valid as
  // one.
  REVERB_EXPECT_OK(writer_->CreateItem("table", 1.0,
                                       MakeTrajectory({
                                           {step_[0], step_[0]},
                                           {step_[1]},
                                           {step_[1], step_[1]},
                                       })));
}

TEST_F(TrajectoryWriterSignatureValidationTest, WrongNumColumns) {
  auto status = writer_->CreateItem("table", 1.0,
                                    MakeTrajectory({
                                        {step_[0], step_[0]},
                                        {step_[1]},
                                    }));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr(
          "Unable to create item in table 'table' since the provided "
          "trajectory is inconsistent with the table signature. The "
          "trajectory has 2 columns but the table signature has 3 columns."));
}

TEST_F(TrajectoryWriterSignatureValidationTest, NotFoundTable) {
  auto status = writer_->CreateItem("not_found", 1.0,
                                    MakeTrajectory({
                                        {step_[0], step_[0]},
                                        {step_[1]},
                                        {step_[1]},
                                    }));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("Unable to create item in table 'not_found' "
                                   "since the table could not be found."));
}

TEST_F(TrajectoryWriterSignatureValidationTest, WrongDtype) {
  auto status = writer_->CreateItem("table", 1.0,
                                    MakeTrajectory({
                                        {step_[0], step_[0]},
                                        {step_[2]},
                                        {step_[1]},
                                    }));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Unable to create item in table 'table' since the provided "
                  "trajectory is inconsistent with the table signature. The "
                  "table expects column 1 to be a float [1] tensor but got a "
                  "double [1] tensor."));
}

TEST_F(TrajectoryWriterSignatureValidationTest, WrongBatchDim) {
  auto status = writer_->CreateItem("table", 1.0,
                                    MakeTrajectory({
                                        {step_[0]},
                                        {step_[1]},
                                        {step_[1]},
                                    }));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(absl::StrFormat(
                  "Unable to create item in table 'table' since the provided "
                  "trajectory is inconsistent with the table signature. The "
                  "table expects column 0 to be a %s [2] tensor but got a "
                  "%s [1] tensor.",
                  Int32Str(), Int32Str())));
}

TEST_F(TrajectoryWriterSignatureValidationTest, WrongElementShape) {
  auto status = writer_->CreateItem("table", 1.0,
                                    MakeTrajectory({
                                        {step_[0], step_[0]},
                                        {step_[1]},
                                        {step_[3]},
                                    }));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Unable to create item in table 'table' since the provided "
                  "trajectory is inconsistent with the table signature. The "
                  "table expects column 2 to be a float [?] tensor but got a "
                  "float [1,2,2] tensor."));
}

TEST_F(TrajectoryWriterSignatureValidationTest,
       ErrorMessageIncludesTableSignature) {
  auto status = writer_->CreateItem("table", 1.0,
                                    MakeTrajectory({
                                        {step_[0], step_[0]},
                                        {step_[1]},
                                        {step_[3]},
                                    }));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(absl::StrFormat(
                  "\n\nThe table signature is:\n\t"
                  "0: Tensor<name: 'first_col', dtype: %s, shape: [2]>, "
                  "1: Tensor<name: 'second_col', dtype: float, shape: [1]>, "
                  "2: Tensor<name: 'var_length_col', dtype: float, shape: [?]>",
                  Int32Str())));
}

TEST_F(TrajectoryWriterSignatureValidationTest,
       ErrorMessageIncludesTrajectorySignature) {
  auto status = writer_->CreateItem("table", 1.0,
                                    MakeTrajectory({
                                        {step_[0], step_[0]},
                                        {step_[1]},
                                        {step_[3]},
                                    }));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(absl::StrFormat(
                  "\n\nThe provided trajectory signature is:\n\t"
                  "0: Tensor<name: '0', dtype: %s, shape: [2]>, "
                  "1: Tensor<name: '1', dtype: float, shape: [1]>, "
                  "2: Tensor<name: '2', dtype: float, shape: [1,2,2]>",
                  Int32Str())));
}

TEST(TrajectoryWriter, EndEpisodeCanClearBuffers) {
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  AsyncInterface async;
  EXPECT_CALL(*stub, async()).WillRepeatedly(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2));

  // Take a step.
  StepRef step;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));

  // If we don't clear the buffer then the reference should be alive after.
  REVERB_ASSERT_OK(writer.EndEpisode(/*clear_buffers=*/false));
  EXPECT_FALSE(step[0]->expired());

  // If we clear the buffer then the reference should expire.
  REVERB_ASSERT_OK(writer.EndEpisode(/*clear_buffers=*/true));
  EXPECT_TRUE(step[0]->expired());
}

TEST(TrajectoryWriter, EndEpisodeFinalizesChunksEvenIfNoItemReferenceIt) {
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  AsyncInterface async;
  EXPECT_CALL(*stub, async()).WillRepeatedly(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2));

  // Take a step.
  StepRef step;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));

  // The chunk is not yet finalized as `max_chunk_length` is 2.
  EXPECT_FALSE(step[0]->lock()->IsReady());

  // Calling `EndEpisode` should trigger the finalization of the chunk even if
  // it is not used by any item. Note that this is different from Flush which
  // only finalizes chunks which owns `CellRef`s that are referenced by pending
  // items.
  REVERB_ASSERT_OK(writer.EndEpisode(/*clear_buffers=*/false));
  EXPECT_TRUE(step[0]->lock()->IsReady());
}

TEST(TrajectoryWriter, EndEpisodeResetsEpisodeKeyAndStep) {
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  AsyncInterface async;
  EXPECT_CALL(*stub, async()).WillRepeatedly(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/2));

  // Take two steps in two different episodes.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));

  REVERB_ASSERT_OK(writer.EndEpisode(/*clear_buffers=*/false));

  StepRef second;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &second));

  // Verify that the `episode_key` was changed between episodes and that the
  // episode step was reset to 0.
  EXPECT_NE(first[0]->lock()->episode_id(), second[0]->lock()->episode_id());
  EXPECT_EQ(first[0]->lock()->episode_step(), 0);
  EXPECT_EQ(second[0]->lock()->episode_step(), 0);
}

TEST(TrajectoryWriter, EpisodeStepIsIncrementedByAppend) {
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  AsyncInterface async;
  EXPECT_CALL(*stub, async()).WillRepeatedly(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/2));

  // The counter should start at 0.
  EXPECT_EQ(writer.episode_steps(), 0);

  // Each append call should increment the counter.
  for (int i = 1; i < 11; i++) {
    StepRef step;
    REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));
    EXPECT_EQ(writer.episode_steps(), i);
  }

  // AppendPartial should not have any impact on the counter.
  {
    StepRef step;
    REVERB_ASSERT_OK(writer.AppendPartial(Step({MakeTensor(kIntSpec)}), &step));
  }
  EXPECT_EQ(writer.episode_steps(), 10);

  // Closing the partial step should increment it though.
  {
    StepRef step;
    REVERB_ASSERT_OK(writer.Append(Step({}), &step));
  }
  EXPECT_EQ(writer.episode_steps(), 11);
}

TEST(TrajectoryWriter, EpisodeStepIsNotIncrementedByAppendPartial) {
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  AsyncInterface async;
  EXPECT_CALL(*stub, async()).WillRepeatedly(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/2));

  // The counter should start at 0.
  EXPECT_EQ(writer.episode_steps(), 0);
  // AppendPartial should not have any impact on the counter.
  {
    StepRef step;
    REVERB_ASSERT_OK(writer.AppendPartial(Step({MakeTensor(kIntSpec)}), &step));
  }
  EXPECT_EQ(writer.episode_steps(), 0);

  // Closing the partial step should increment it though.
  {
    StepRef step;
    REVERB_ASSERT_OK(writer.Append(Step({}), &step));
  }
  EXPECT_EQ(writer.episode_steps(), 1);
}

TEST(TrajectoryWriter, EpisodeStepIsResetByEndEpisode) {
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  AsyncInterface async;
  EXPECT_CALL(*stub, async()).WillRepeatedly(Return(&async));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/2));

  // Take a step and check that the counter has been incremented.
  {
    StepRef step;
    REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));
  }
  EXPECT_EQ(writer.episode_steps(), 1);

  // Ending the episode while clearing the buffers should reset the counter.
  REVERB_ASSERT_OK(writer.EndEpisode(true, absl::Milliseconds(100)));
  EXPECT_EQ(writer.episode_steps(), 0);

  // Repeat the process but don't clear the buffers when ending the episode.
  {
    StepRef step;
    REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));
  }
  EXPECT_EQ(writer.episode_steps(), 1);
  REVERB_ASSERT_OK(writer.EndEpisode(false, absl::Milliseconds(100)));
  EXPECT_EQ(writer.episode_steps(), 0);
}

TEST(TrajectoryWriter, EndEpisodeReturnsIfTimeoutExpired) {
  AsyncInterface fail_stream(false);
  auto stub = std::make_shared<MockReverbServiceAsyncStub>();
  EXPECT_CALL(*stub, async())
     .WillRepeatedly(Return(&fail_stream));

  TrajectoryWriter writer(
      stub, MakeOptions(/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2));

  // Create an item.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}})));
  // EndEpisode will not be able to complete and thus should timeout.
  auto status = writer.EndEpisode(true, absl::Milliseconds(100));
  fail_stream.stream_.BlockUntilNumRequestsIs(1);
  fail_stream.stream_.SetStatus(grpc::Status(grpc::StatusCode::INTERNAL, ""));
  writer.OnReadDone(false);

  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr("Timeout exceeded"));

  // Close the writer to avoid having to mock the item confirmation response.
  writer.Close();
}

class TrajectoryWriterOptionsTest : public ::testing::Test {
 protected:
  void ExpectInvalidArgumentWithMessage(const std::string& message) {
    auto status = options_.Validate();
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(std::string(status.message()), ::testing::HasSubstr(message));
  }

  TrajectoryWriter::Options options_;
};

TEST_F(TrajectoryWriterOptionsTest, Valid) {
  options_ = MakeOptions(/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2);
  REVERB_EXPECT_OK(options_.Validate());
}

TEST_F(TrajectoryWriterOptionsTest, NoChunkerOptions) {
  options_.chunker_options = nullptr;
  ExpectInvalidArgumentWithMessage("chunker_options must be set.");
}

TEST_F(TrajectoryWriterOptionsTest, ZeroMaxChunkLength) {
  options_ = MakeOptions(/*max_chunk_length=*/0, /*num_keep_alive_refs=*/2);
  ExpectInvalidArgumentWithMessage("max_chunk_length must be > 0 but got 0.");
}

TEST_F(TrajectoryWriterOptionsTest, NegativeMaxChunkLength) {
  options_ = MakeOptions(/*max_chunk_length=*/-1, /*num_keep_alive_refs=*/2);
  ExpectInvalidArgumentWithMessage("max_chunk_length must be > 0 but got -1.");
}

TEST_F(TrajectoryWriterOptionsTest, ZeroNumKeepAliveRefs) {
  options_ = MakeOptions(/*max_chunk_length=*/2, /*num_keep_alive_refs=*/0);
  ExpectInvalidArgumentWithMessage(
      "num_keep_alive_refs must be > 0 but got 0.");
}

TEST_F(TrajectoryWriterOptionsTest, NegativeNumKeepAliveRefs) {
  options_ = MakeOptions(/*max_chunk_length=*/2, /*num_keep_alive_refs=*/-1);
  ExpectInvalidArgumentWithMessage(
      "num_keep_alive_refs must be > 0 but got -1.");
}

TEST_F(TrajectoryWriterOptionsTest, NumKeepAliveLtMaxChunkLength) {
  options_ = MakeOptions(/*max_chunk_length=*/6, /*num_keep_alive_refs=*/5);
  ExpectInvalidArgumentWithMessage(
      "num_keep_alive_refs (5) must be >= max_chunk_length (6).");
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
