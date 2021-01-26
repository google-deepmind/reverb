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

#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include "grpcpp/test/mock_stream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/reverb_service_mock.grpc.pb.h"
#include "reverb/cc/support/queue.h"
#include "reverb/cc/support/signature.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace deepmind {
namespace reverb {
namespace {

using ::grpc::testing::MockClientReaderWriter;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Return;

using Step = ::std::vector<::absl::optional<::tensorflow::Tensor>>;
using StepRef = ::std::vector<::absl::optional<::std::weak_ptr<CellRef>>>;
using TrajectoryRef = ::std::vector<::std::vector<::std::weak_ptr<CellRef>>>;

const auto kIntSpec = internal::TensorSpec{"0", tensorflow::DT_INT32, {1}};
const auto kFloatSpec = internal::TensorSpec{"0", tensorflow::DT_FLOAT, {1}};

MATCHER(IsChunk, "") { return arg.has_chunk(); }

MATCHER(IsItem, "") { return arg.item().send_confirmation(); }

inline std::string Int32Str() {
  return tensorflow::DataTypeString(tensorflow::DT_INT32);
}

inline tensorflow::Tensor MakeTensor(const internal::TensorSpec& spec) {
  tensorflow::TensorShape shape;
  REVERB_CHECK(spec.shape.AsTensorShape(&shape));
  tensorflow::Tensor tensor(spec.dtype, shape);
  return tensor;
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
    pending_confirmation_.Close();
    return grpc::Status::OK;
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

TEST(CellRef, IsReady) {
  Chunker chunker(kIntSpec, 2, 5);

  std::weak_ptr<CellRef> ref;
  TF_ASSERT_OK(chunker.Append(MakeTensor(kIntSpec), &ref));

  // Chunk is not finalized yet.
  EXPECT_FALSE(ref.lock()->IsReady());

  // Force chunk creation.
  TF_ASSERT_OK(chunker.Flush());
  EXPECT_TRUE(ref.lock()->IsReady());
}

TEST(Chunker, AppendValidatesSpecDtype) {
  Chunker chunker(kIntSpec, /*max_chunk_length=*/2, /*num_keep_alive_refs=*/5);

  std::weak_ptr<CellRef> ref;
  auto status = chunker.Append(MakeTensor(kFloatSpec), &ref);

  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
              testing::HasSubstr(
                  absl::StrCat("Tensor of wrong dtype provided for column 0. "
                               "Got float but expected ",
                               Int32Str(), ".")));
}

TEST(Chunker, AppendValidatesSpecShape) {
  Chunker chunker(kIntSpec, /*max_chunk_length=*/2, /*num_keep_alive_refs=*/5);

  std::weak_ptr<CellRef> ref;
  auto status = chunker.Append(
      MakeTensor(internal::TensorSpec{kIntSpec.name, kIntSpec.dtype, {2}}),
      &ref);

  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(
      status.error_message(),
      testing::HasSubstr("Tensor of incompatible shape provided for column 0. "
                         "Got [2] which is incompatible with [1]."));
}

TEST(Chunker, AppendFlushesOnMaxChunkLength) {
  Chunker chunker(kIntSpec, /*max_chunk_length=*/2, /*num_keep_alive_refs=*/5);

  // Buffer is not full after first step.
  std::weak_ptr<CellRef> first;
  TF_ASSERT_OK(chunker.Append(MakeTensor(kIntSpec), &first));
  EXPECT_FALSE(first.lock()->IsReady());

  // Second step should trigger flushing of buffer.
  std::weak_ptr<CellRef> second;
  TF_ASSERT_OK(chunker.Append(MakeTensor(kIntSpec), &second));
  EXPECT_TRUE(first.lock()->IsReady());
  EXPECT_TRUE(second.lock()->IsReady());
}

TEST(Chunker, Flush) {
  Chunker chunker(kIntSpec, /*max_chunk_length=*/2, /*num_keep_alive_refs=*/5);
  std::weak_ptr<CellRef> ref;
  TF_ASSERT_OK(chunker.Append(MakeTensor(kIntSpec), &ref));
  EXPECT_FALSE(ref.lock()->IsReady());
  TF_ASSERT_OK(chunker.Flush());
  EXPECT_TRUE(ref.lock()->IsReady());
}

TEST(Chunker, DeletesRefsWhenMageAgeExceeded) {
  Chunker chunker(kIntSpec, /*max_chunk_length=*/2, /*num_keep_alive_refs=*/3);

  std::weak_ptr<CellRef> first;
  TF_ASSERT_OK(chunker.Append(MakeTensor(kIntSpec), &first));
  EXPECT_FALSE(first.expired());

  std::weak_ptr<CellRef> second;
  TF_ASSERT_OK(chunker.Append(MakeTensor(kIntSpec), &second));
  EXPECT_FALSE(first.expired());
  EXPECT_FALSE(second.expired());

  std::weak_ptr<CellRef> third;
  TF_ASSERT_OK(chunker.Append(MakeTensor(kIntSpec), &third));
  EXPECT_FALSE(first.expired());
  EXPECT_FALSE(second.expired());
  EXPECT_FALSE(third.expired());

  std::weak_ptr<CellRef> fourth;
  TF_ASSERT_OK(chunker.Append(MakeTensor(kIntSpec), &fourth));
  EXPECT_TRUE(first.expired());
  EXPECT_FALSE(second.expired());
  EXPECT_FALSE(third.expired());
  EXPECT_FALSE(fourth.expired());
}

TEST(Chunker, GetKeepKeys) {
  Chunker chunker(kIntSpec, /*max_chunk_length=*/2, /*num_keep_alive_refs=*/2);

  std::weak_ptr<CellRef> first;
  TF_ASSERT_OK(chunker.Append(MakeTensor(kIntSpec), &first));
  EXPECT_THAT(chunker.GetKeepKeys(), ElementsAre(first.lock()->chunk_key()));

  // The second ref will belong to the same chunk.
  std::weak_ptr<CellRef> second;
  TF_ASSERT_OK(chunker.Append(MakeTensor(kIntSpec), &second));
  EXPECT_THAT(chunker.GetKeepKeys(), ElementsAre(first.lock()->chunk_key()));

  // The third ref will belong to a new chunk. The first ref is now expired but
  // since the second ref belong to the same chunk we expect the chunker to tell
  // us to keep both chunks around.
  std::weak_ptr<CellRef> third;
  TF_ASSERT_OK(chunker.Append(MakeTensor(kIntSpec), &third));
  EXPECT_THAT(chunker.GetKeepKeys(), ElementsAre(second.lock()->chunk_key(),
                                                 third.lock()->chunk_key()));

  // Adding a fourth value results in the second one expiring. The only chunk
  // which should be kept thus is the one referenced by the third and fourth.
  std::weak_ptr<CellRef> fourth;
  TF_ASSERT_OK(chunker.Append(MakeTensor(kIntSpec), &fourth));
  EXPECT_THAT(chunker.GetKeepKeys(), ElementsAre(third.lock()->chunk_key()));
}

TEST(TrajectoryWriter, AppendValidatesDtype) {
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_))
      .WillOnce(Return(new MockClientReaderWriter<InsertStreamRequest,
                                                  InsertStreamResponse>()));

  TrajectoryWriter writer(
      stub, {/*max_chunk_length=*/10, /*num_keep_alive_refs=*/10});
  StepRef refs;

  // Initiate the spec with the first step.
  TF_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kFloatSpec)}), &refs));

  // Change the dtypes in the next step.
  auto status =
      writer.Append(Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}), &refs);
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
              testing::HasSubstr(
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
  TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &refs));

  // Change the dtypes in the next step.
  auto status = writer.Append(Step({MakeTensor(internal::TensorSpec{
                                  kIntSpec.name, kIntSpec.dtype, {3}})}),
                              &refs);
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(
      status.error_message(),
      testing::HasSubstr("Tensor of incompatible shape provided for column 0. "
                         "Got [3] which is incompatible with [1]."));
}

TEST(TrajectoryWriter, AppendAcceptsPartialSteps) {
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(new FakeStream()));

  TrajectoryWriter writer(
      stub, {/*max_chunk_length=*/10, /*num_keep_alive_refs=*/10});

  // Initiate the spec with the first step.
  StepRef both;
  TF_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kFloatSpec)}), &both));

  // Only append to the first column.
  StepRef first_column_only;
  TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec), absl::nullopt}),
                             &first_column_only));
  EXPECT_FALSE(first_column_only[1].has_value());
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
    TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &refs));
  }
}

TEST(TrajectoryWriter, ItemSentStraightAwayIfChunksReady) {
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1});
  StepRef refs;
  TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &refs));

  // Nothing sent before the item created.
  EXPECT_THAT(stream->requests(), testing::IsEmpty());

  // The chunk is completed so inserting an item should result in both chunk
  // and item being sent.
  TF_ASSERT_OK(
      writer.InsertItem("table", 1.0, TrajectoryRef{{refs[0].value()}}));

  while (stream->requests().size() < 2) {
  }

  // Chunk is sent before item.
  EXPECT_THAT(stream->requests(), ElementsAre(IsChunk(), IsItem()));

  // Adding a second item should result in the item being sent straight away.
  // Note that the chunk is not sent again.
  TF_ASSERT_OK(
      writer.InsertItem("table", 0.5, TrajectoryRef({{refs[0].value()}})));
  while (stream->requests().size() < 3) {
  }
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
  TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}),
                             &first));

  // Create an item which references the first row in the two columns.
  TF_ASSERT_OK(writer.InsertItem("table", 1.0,
                                 {{first[0].value()}, {first[1].value()}}));

  // No data is sent yet since the chunks are not completed.
  EXPECT_THAT(stream->requests(), testing::IsEmpty());

  // In the second step we only write to the first column. This should trigger
  // the transmission of the first chunk but not the item as it needs to wait
  // for the chunk in the second column to be completed.
  StepRef second;
  TF_ASSERT_OK(
      writer.Append(Step({MakeTensor(kIntSpec), absl::nullopt}), &second));
  while (stream->requests().empty()) {
  }
  EXPECT_THAT(stream->requests(), ElementsAre(IsChunk()));

  // Writing to the first column again, even if we do it twice and trigger a new
  // chunk to be completed, should not trigger any new messages.
  for (int i = 0; i < 2; i++) {
    StepRef refs;
    TF_ASSERT_OK(
        writer.Append(Step({MakeTensor(kIntSpec), absl::nullopt}), &refs));
  }
  EXPECT_THAT(stream->requests(), testing::SizeIs(1));

  // Writing to the second column will trigger the completion of the chunk in
  // the second column. This in turn should trigger the transmission of the new
  // chunk and the item.
  StepRef third;
  TF_ASSERT_OK(
      writer.Append(Step({absl::nullopt, MakeTensor(kIntSpec)}), &third));
  while (stream->requests().size() < 3) {
  }
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
  TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}),
                             &first));

  // Create an item which references the first row in second column.
  TF_ASSERT_OK(writer.InsertItem("table", 1.0, {{first[1].value()}}));

  // No data is sent yet since the chunks are not completed.
  EXPECT_THAT(stream->requests(), testing::IsEmpty());

  // Calling flush should trigger the chunk creation of the second column only.
  // Since the first column isn't referenced by the pending item there is no
  // need for it to be prematurely finalized. Since all chunks required by the
  // pending item is now ready, the chunk and the item should be sent to the
  // server.
  TF_ASSERT_OK(writer.Flush());
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
    TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));

    // Create an item which references the first row in the incomplete chunk..
    TF_ASSERT_OK(writer.InsertItem("table", 1.0, {{first[0].value()}}));

    // No data is sent yet since the chunks are not completed.
    EXPECT_THAT(stream->requests(), testing::IsEmpty());
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
  TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  TF_ASSERT_OK(writer.InsertItem("table", 1.0, {{first[0].value()}}));
  TF_ASSERT_OK(writer.Flush());

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
  TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  TF_ASSERT_OK(writer.InsertItem("table", 1.0, {{first[0].value()}}));

  // Flushing should return the error encountered by the stream worker.
  auto flush_status = writer.Flush();
  EXPECT_EQ(flush_status.code(), tensorflow::error::INTERNAL);
  EXPECT_THAT(flush_status.error_message(), testing::HasSubstr("A reason"));

  // The same error should be encountered in all methods.
  auto insert_status = writer.InsertItem("table", 1.0, {{first[0].value()}});
  EXPECT_EQ(insert_status.code(), tensorflow::error::INTERNAL);
  EXPECT_THAT(insert_status.error_message(), testing::HasSubstr("A reason"));

  auto append_status = writer.Append(Step({MakeTensor(kIntSpec)}), &first);
  EXPECT_EQ(append_status.code(), tensorflow::error::INTERNAL);
  EXPECT_THAT(append_status.error_message(), testing::HasSubstr("A reason"));
}

TEST(TrajectoryWriter, FlushReturnsIfTimeoutExpired) {
  absl::Notification write_block;
  auto* stream =
      new MockClientReaderWriter<InsertStreamRequest, InsertStreamResponse>();
  EXPECT_CALL(*stream, Write(_, _))
      .WillOnce(testing::Invoke([&](auto, auto) {
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
  TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  TF_ASSERT_OK(writer.InsertItem("table", 1.0, {{first[0].value()}}));

  // Flushing should return the error encountered by the stream worker.
  auto status = writer.Flush(absl::Milliseconds(100));
  EXPECT_EQ(status.code(), tensorflow::error::DEADLINE_EXCEEDED);
  EXPECT_THAT(status.error_message(),
              testing::HasSubstr("Timeout exceeded with 1 items waiting to be "
                                 "written and 0 items awaiting confirmation."));

  // Unblock the writer.
  write_block.Notify();

  // Close the writer to avoid having to mock the item confirmation response.
  writer.Close();
}

TEST(TrajectoryWriter, InsertItemRejectsExpiredCellRefs) {
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(new FakeStream()));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1});

  // Take two steps.
  StepRef first;
  StepRef second;
  TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &second));

  // The num_keep_alive_refs is set to 1 so the first step has expired.
  auto status = writer.InsertItem("table", 1.0, {{first[0].value()}});
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
              testing::HasSubstr("Trajectory contains expired CellRef."));
}

TEST(TrajectoryWriter, KeepKeysOnlyIncludesStreamedKeys) {
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/1});

  // Create a step with two columns.
  StepRef first;
  TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}),
                             &first));

  // Create an item which only references one of the columns.
  TF_ASSERT_OK(writer.InsertItem("table", 1.0, {{first[0].value()}}));
  TF_ASSERT_OK(writer.Flush());

  // Only the chunk of the first column has been used (and thus streamed). The
  // server should thus only be instructed to keep the one chunk around.
  EXPECT_THAT(stream->requests(), ElementsAre(IsChunk(), IsItem()));
  EXPECT_THAT(stream->requests()[1].item().keep_chunk_keys(),
              ElementsAre(first[0].value().lock()->chunk_key()));
}

TEST(TrajectoryWriter, KeepKeysOnlyIncludesLiveChunks) {
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub,
                          {/*max_chunk_length=*/1, /*num_keep_alive_refs=*/2});

  // Take a step and insert a trajectory.
  StepRef first;
  TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  TF_ASSERT_OK(writer.InsertItem("table", 1.0, {{first[0].value()}}));
  TF_ASSERT_OK(writer.Flush());

  // The one chunk that has been sent should be kept alive.
  EXPECT_THAT(stream->requests().back().item().keep_chunk_keys(),
              ElementsAre(first[0].value().lock()->chunk_key()));

  // Take a second step and insert a trajectory.
  StepRef second;
  TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &second));
  TF_ASSERT_OK(writer.InsertItem("table", 1.0, {{second[0].value()}}));
  TF_ASSERT_OK(writer.Flush());

  // Both chunks should be kept alive since num_keep_alive_refs is 2.
  EXPECT_THAT(stream->requests().back().item().keep_chunk_keys(),
              ElementsAre(first[0].value().lock()->chunk_key(),
                          second[0].value().lock()->chunk_key()));

  // Take a third step and insert a trajectory.
  StepRef third;
  TF_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &third));
  TF_ASSERT_OK(writer.InsertItem("table", 1.0, {{third[0].value()}}));
  TF_ASSERT_OK(writer.Flush());

  // The chunk of the first step has now expired and thus the server no longer
  // need to keep it alive.
  EXPECT_THAT(stream->requests().back().item().keep_chunk_keys(),
              ElementsAre(second[0].value().lock()->chunk_key(),
                          third[0].value().lock()->chunk_key()));
}

TEST(TrajectoryWriter, InsertItemValidatesTrajectoryDtype) {
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub, {/*max_chunk_length=*/1, /*max_age=*/2});

  // Take a step with two columns with different dtypes.
  StepRef step;
  TF_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kFloatSpec)}), &step));

  // Create a trajectory where the two dtypes are used in the same column.
  auto status =
      writer.InsertItem("table", 1.0, {{step[0].value(), step[1].value()}});
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
              testing::HasSubstr(absl::StrCat(
                  "Column 0 references tensors with different dtypes: ",
                  Int32Str(), " (index 0) != float (index 1).")));
}

TEST(TrajectoryWriter, InsertItemValidatesTrajectoryShapes) {
  auto* stream = new FakeStream();
  auto stub = std::make_shared</* grpc_gen:: */MockReverbServiceStub>();
  EXPECT_CALL(*stub, InsertStreamRaw(_)).WillOnce(Return(stream));

  TrajectoryWriter writer(stub, {/*max_chunk_length=*/1, /*max_age=*/2});

  // Take a step with two columns with different shapes.
  StepRef step;

  TF_ASSERT_OK(writer.Append(
      Step({
          MakeTensor(kIntSpec),
          MakeTensor(internal::TensorSpec{"1", kIntSpec.dtype, {2}}),
      }),
      &step));

  // Create a trajectory where the two shapes are used in the same column.
  auto status =
      writer.InsertItem("table", 1.0, {{step[0].value(), step[1].value()}});
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
              testing::HasSubstr(
                  "Column 0 references tensors with incompatible shapes: [1] "
                  "(index 0) not compatible with [2] (index 1)."));
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
