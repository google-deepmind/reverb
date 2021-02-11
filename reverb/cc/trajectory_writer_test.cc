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

#include <memory>
#include <string>

#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include "grpcpp/test/mock_stream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/reverb_service_mock.grpc.pb.h"
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

using ::grpc::testing::MockClientReaderWriter;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Return;
using ::testing::UnorderedElementsAre;

using Step = ::std::vector<::absl::optional<::tensorflow::Tensor>>;
using StepRef = ::std::vector<::absl::optional<::std::weak_ptr<CellRef>>>;

const auto kIntSpec = internal::TensorSpec{"0", tensorflow::DT_INT32, {1}};
const auto kFloatSpec = internal::TensorSpec{"0", tensorflow::DT_FLOAT, {1}};

MATCHER(IsChunk, "") { return arg.has_chunk(); }

MATCHER(IsItem, "") { return arg.item().send_confirmation(); }

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

class FakeStream
    : public MockClientReaderWriter<InsertStreamRequest, InsertStreamResponse> {
 public:
  FakeStream()
      : requests_(std::make_shared<std::vector<InsertStreamRequest>>()),
        pending_confirmation_(10) {}

  ~FakeStream() { pending_confirmation_.Close(); }

  bool Write(const InsertStreamRequest& msg,
             grpc::WriteOptions options) override {
    absl::MutexLock lock(&mu_);
    requests_->push_back(msg);

    if (msg.item().send_confirmation()) {
      REVERB_CHECK(pending_confirmation_.Push(msg.item().item().key()));
    }

    return true;
  }

  bool Read(InsertStreamResponse* response) override {
    uint64_t confirm_id;
    if (!pending_confirmation_.Pop(&confirm_id)) {
      return false;
    }
    response->set_key(confirm_id);
    return true;
  }

  grpc::Status Finish() override {
    absl::MutexLock lock(&mu_);
    pending_confirmation_.Close();
    return grpc::Status::OK;
  }

  void BlockUntilNumRequestsIs(int size) const {
    absl::MutexLock lock(&mu_);
    auto trigger = [size, this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return requests_->size() == size;
    };
    mu_.Await(absl::Condition(&trigger));
  }

  const std::vector<InsertStreamRequest>& requests() const {
    absl::MutexLock lock(&mu_);
    return *requests_;
  }

  std::shared_ptr<std::vector<InsertStreamRequest>> requests_ptr() const {
    absl::MutexLock lock(&mu_);
    return requests_;
  }

 private:
  mutable absl::Mutex mu_;
  std::shared_ptr<std::vector<InsertStreamRequest>> requests_
      ABSL_GUARDED_BY(mu_);
  internal::Queue<uint64_t> pending_confirmation_;
};

TEST(TrajectoryWriter, AppendValidatesDtype) {
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_))
      .WillOnce(Return(new MockClientReaderWriter<InsertStreamRequest,
                                                  InsertStreamResponse>()));

  TrajectoryWriter writer(
      stub, {/*max_chunk_length=*/10, /*num_keep_alive_refs=*/10});
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
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(new FakeStream()));

  TrajectoryWriter writer(
      stub, {/*max_chunk_length=*/10, /*num_keep_alive_refs=*/10});
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
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(new FakeStream()));

  TrajectoryWriter writer(
      stub, {/*max_chunk_length=*/10, /*num_keep_alive_refs=*/10});

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

TEST(TrajectoryWriter, ConfigureChunkerOnExistingColumn) {
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(new FakeStream()));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1});

  // Create the column with the first step.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));

  // The chunk should be created automatically since max_chunk_length is 1.
  EXPECT_TRUE(first[0]->lock()->IsReady());

  // Reconfigure the column to have a chunk length of 2 instead.
  REVERB_ASSERT_OK(writer.ConfigureChunker(
      0, {/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2}));

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
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(new FakeStream()));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1});

  // Create the first column with the first step.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));

  // The chunk should be created automatically since max_chunk_length is 1.
  EXPECT_TRUE(first[0]->lock()->IsReady());

  // Configure the second column (not yet seen) to have max_chunk_length 2
  // instead of 1.
  REVERB_ASSERT_OK(writer.ConfigureChunker(
      1, {/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2}));

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
  auto* stream = new FakeStream();
  EXPECT_CALL(*stream, Write(_, _)).Times(0);

  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1});
  StepRef refs;

  for (int i = 0; i < 10; ++i) {
    REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &refs));
  }
}

TEST(TrajectoryWriter, ItemSentStraightAwayIfChunksReady) {
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub, {/*max_chunk_length=*/1,
                                 /*num_keep_alive_refs=*/1});
  StepRef refs;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &refs));

  // Nothing sent before the item created.
  EXPECT_THAT(stream->requests(), ::testing::IsEmpty());

  // The chunk is completed so inserting an item should result in both chunk
  // and item being sent.
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{refs[0]}})));

  stream->BlockUntilNumRequestsIs(2);

  // Chunk is sent before item.
  EXPECT_THAT(stream->requests(), ElementsAre(IsChunk(), IsItem()));

  // Adding a second item should result in the item being sent straight away.
  // Note that the chunk is not sent again.
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 0.5, MakeTrajectory({{refs[0]}})));

  stream->BlockUntilNumRequestsIs(3);

  EXPECT_THAT(stream->requests()[2], IsItem());
}

TEST(TrajectoryWriter, ItemIsSentWhenAllChunksDone) {
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2});

  // Write to both columns in the first step.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}), &first));

  // Create an item which references the first row in the two columns.
  REVERB_ASSERT_OK(writer.CreateItem("table", 1.0,
                                     MakeTrajectory({{first[0]}, {first[1]}})));

  // No data is sent yet since the chunks are not completed.
  EXPECT_THAT(stream->requests(), ::testing::IsEmpty());

  // In the second step we only write to the first column. This should trigger
  // the transmission of the first chunk but not the item as it needs to wait
  // for the chunk in the second column to be completed.
  StepRef second;
  REVERB_ASSERT_OK(
      writer.Append(Step({MakeTensor(kIntSpec), absl::nullopt}), &second));

  stream->BlockUntilNumRequestsIs(1);

  EXPECT_THAT(stream->requests(), ElementsAre(IsChunk()));

  // Writing to the first column again, even if we do it twice and trigger a new
  // chunk to be completed, should not trigger any new messages.
  for (int i = 0; i < 2; i++) {
    StepRef refs;
    REVERB_ASSERT_OK(
        writer.Append(Step({MakeTensor(kIntSpec), absl::nullopt}), &refs));
  }
  EXPECT_THAT(stream->requests(), ::testing::SizeIs(1));

  // Writing to the second column will trigger the completion of the chunk in
  // the second column. This in turn should trigger the transmission of the new
  // chunk and the item.
  StepRef third;
  REVERB_ASSERT_OK(
      writer.Append(Step({absl::nullopt, MakeTensor(kIntSpec)}), &third));

  stream->BlockUntilNumRequestsIs(3);

  EXPECT_THAT(stream->requests(), ElementsAre(IsChunk(), IsChunk(), IsItem()));
}

TEST(TrajectoryWriter, FlushSendsPendingItems) {
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2});

  // Write to both columns in the first step.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}), &first));

  // Create an item which references the first row in second column.
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[1]}})));

  // No data is sent yet since the chunks are not completed.
  EXPECT_THAT(stream->requests(), ::testing::IsEmpty());

  // Calling flush should trigger the chunk creation of the second column only.
  // Since the first column isn't referenced by the pending item there is no
  // need for it to be prematurely finalized. Since all chunks required by the
  // pending item is now ready, the chunk and the item should be sent to the
  // server.
  REVERB_ASSERT_OK(writer.Flush());
  EXPECT_FALSE(first[0].value().lock()->IsReady());
  EXPECT_TRUE(first[1].value().lock()->IsReady());
  EXPECT_THAT(stream->requests(), ElementsAre(IsChunk(), IsItem()));
}

TEST(TrajectoryWriter, DestructorFlushesPendingItems) {
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  // The requests vector needs to outlive the stream.
  auto requests = stream->requests_ptr();
  {
    TrajectoryWriter writer(
        stub, {/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2});

    // Write to both columns in the first step.
    StepRef first;
    REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));

    // Create an item which references the first row in the incomplete chunk..
    REVERB_ASSERT_OK(
        writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}})));

    // No data is sent yet since the chunks are not completed.
    EXPECT_THAT(stream->requests(), ::testing::IsEmpty());
  }

  EXPECT_THAT(*requests, ElementsAre(IsChunk(), IsItem()));
}

TEST(TrajectoryWriter, RetriesOnTransientError) {
  auto* fail_stream =
      new MockClientReaderWriter<InsertStreamRequest, InsertStreamResponse>();
  EXPECT_CALL(*fail_stream, Write(IsChunk(), _)).WillOnce(Return(true));
  EXPECT_CALL(*fail_stream, Write(IsItem(), _)).WillOnce(Return(false));
  EXPECT_CALL(*fail_stream, Read(_)).WillOnce(Return(false));
  EXPECT_CALL(*fail_stream, Finish())
      .WillOnce(Return(grpc::Status(grpc::StatusCode::UNAVAILABLE, "")));

  auto* success_stream = new FakeStream();

  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_))
      .WillOnce(Return(fail_stream))
      .WillOnce(Return(success_stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1});

  // Create an item and wait for it to be confirmed.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}})));
  REVERB_ASSERT_OK(writer.Flush());

  // The first stream will fail on the second request (item). The writer should
  // then close the stream and once it sees the UNAVAILABLE error open a nee
  // stream. The writer should then proceed to resend the chunk since there is
  // no guarantee that the new stream is connected to the same server and thus
  // the data might not exist on the server.
  EXPECT_THAT(success_stream->requests(), ElementsAre(IsChunk(), IsItem()));
}

TEST(TrajectoryWriter, StopsOnNonTransientError) {
  auto* fail_stream =
      new MockClientReaderWriter<InsertStreamRequest, InsertStreamResponse>();
  EXPECT_CALL(*fail_stream, Write(IsChunk(), _)).WillOnce(Return(true));
  EXPECT_CALL(*fail_stream, Write(IsItem(), _)).WillOnce(Return(false));
  EXPECT_CALL(*fail_stream, Read(_)).WillOnce(Return(false));
  EXPECT_CALL(*fail_stream, Finish())
      .WillOnce(Return(grpc::Status(grpc::StatusCode::INTERNAL, "A reason")));

  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(fail_stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1});

  // Create an item.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}})));

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

TEST(TrajectoryWriter, FlushReturnsIfTimeoutExpired) {
  absl::Notification write_block;
  auto* stream =
      new MockClientReaderWriter<InsertStreamRequest, InsertStreamResponse>();
  EXPECT_CALL(*stream, Write(_, _))
      .WillOnce(::testing::Invoke([&](auto, auto) {
        write_block.WaitForNotification();
        return true;
      }))
      .WillRepeatedly(Return(true));
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1});

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
      ::testing::HasSubstr("Timeout exceeded with 1 items waiting to be "
                           "written and 0 items awaiting confirmation."));

  // Unblock the writer.
  write_block.Notify();

  // Close the writer to avoid having to mock the item confirmation response.
  writer.Close();
}

TEST(TrajectoryWriter, FlushCanIgnorePendingItems) {
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2});

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
  EXPECT_THAT(stream->requests(), ElementsAre(IsChunk(), IsItem()));

  // The chunk of the first item is finalized while the other is not.
  EXPECT_TRUE(first[0]->lock()->IsReady());
  EXPECT_FALSE(first[1]->lock()->IsReady());
}

TEST(TrajectoryWriter, CreateItemRejectsExpiredCellRefs) {
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(new FakeStream()));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1});

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
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1});

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
  EXPECT_THAT(stream->requests(), UnorderedElementsAre(IsChunk(), IsItem()));
  EXPECT_THAT(stream->requests()[1].item().keep_chunk_keys(),
              UnorderedElementsAre(first[0].value().lock()->chunk_key()));
}

TEST(TrajectoryWriter, KeepKeysOnlyIncludesLiveChunks) {
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/2});

  // Take a step and insert a trajectory.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}})));
  REVERB_ASSERT_OK(writer.Flush());

  // The one chunk that has been sent should be kept alive.
  EXPECT_THAT(stream->requests().back().item().keep_chunk_keys(),
              UnorderedElementsAre(first[0].value().lock()->chunk_key()));

  // Take a second step and insert a trajectory.
  StepRef second;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &second));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{second[0]}})));
  REVERB_ASSERT_OK(writer.Flush());

  // Both chunks should be kept alive since num_keep_alive_refs is 2.
  EXPECT_THAT(stream->requests().back().item().keep_chunk_keys(),
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
  EXPECT_THAT(stream->requests().back().item().keep_chunk_keys(),
              UnorderedElementsAre(second[0].value().lock()->chunk_key(),
                                   third[0].value().lock()->chunk_key()));
}

TEST(TrajectoryWriter, CreateItemValidatesTrajectoryDtype) {
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/2});

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
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/2});

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
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1});

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
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1});

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

class TrajectoryWriterSignatureValidationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto* stream = new FakeStream();
    auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
    EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

    LOG(INFO) << tensorflow::PartialTensorShape({}).dims();
    TrajectoryWriter::Options options = {
        .max_chunk_length = 1,
        .num_keep_alive_refs = 1,
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
    writer_ = absl::make_unique<TrajectoryWriter>(stub, options);

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
  }

  std::unique_ptr<TrajectoryWriter> writer_;
  StepRef step_;
};

TEST_F(TrajectoryWriterSignatureValidationTest, Valid) {
  EXPECT_OK(writer_->CreateItem("table", 1.0,
                                MakeTrajectory({
                                    {step_[0], step_[0]},
                                    {step_[1]},
                                    {step_[1]},
                                })));

  // Third column length is undefined so two rows should be just as valid as
  // one.
  EXPECT_OK(writer_->CreateItem("table", 1.0,
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
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2});

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
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2});

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
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/2});

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

TEST(TrajectoryWriter, EndEpisodeReturnsIfTimeoutExpired) {
  absl::Notification write_block;
  auto* stream =
      new MockClientReaderWriter<InsertStreamRequest, InsertStreamResponse>();
  EXPECT_CALL(*stream, Write(_, _))
      .WillOnce(::testing::Invoke([&](auto, auto) {
        write_block.WaitForNotification();
        return true;
      }))
      .WillRepeatedly(Return(true));
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/2, /*num_keep_alive_refs=*/2});

  // Create an item.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}})));

  // EndEpisode will not be able to complete and thus should timeout.
  auto status = writer.EndEpisode(true, absl::Milliseconds(100));
  EXPECT_EQ(status.code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr("Timeout exceeded with 1 items waiting to be "
                           "written and 0 items awaiting confirmation."));

  // Unblock the writer.
  write_block.Notify();

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
  options_.max_chunk_length = 2;
  options_.num_keep_alive_refs = 2;
  REVERB_EXPECT_OK(options_.Validate());
}

TEST_F(TrajectoryWriterOptionsTest, ZeroMaxChunkLength) {
  options_.max_chunk_length = 0;
  options_.num_keep_alive_refs = 2;
  ExpectInvalidArgumentWithMessage("max_chunk_length must be > 0 but got 0.");
}

TEST_F(TrajectoryWriterOptionsTest, NegativeMaxChunkLength) {
  options_.max_chunk_length = -1;
  options_.num_keep_alive_refs = 2;
  ExpectInvalidArgumentWithMessage("max_chunk_length must be > 0 but got -1.");
}

TEST_F(TrajectoryWriterOptionsTest, ZeroNumKeepAliveRefs) {
  options_.max_chunk_length = 2;
  options_.num_keep_alive_refs = 0;
  ExpectInvalidArgumentWithMessage(
      "num_keep_alive_refs must be > 0 but got 0.");
}

TEST_F(TrajectoryWriterOptionsTest, NegativeNumKeepAliveRefs) {
  options_.max_chunk_length = 2;
  options_.num_keep_alive_refs = -1;
  ExpectInvalidArgumentWithMessage(
      "num_keep_alive_refs must be > 0 but got -1.");
}

TEST_F(TrajectoryWriterOptionsTest, NumKeepAliveLtMaxChunkLength) {
  options_.num_keep_alive_refs = 5;
  options_.max_chunk_length = 6;
  ExpectInvalidArgumentWithMessage(
      "num_keep_alive_refs (5) must be >= max_chunk_length (6).");
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
