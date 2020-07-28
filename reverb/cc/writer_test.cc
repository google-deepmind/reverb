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

#include "reverb/cc/writer.h"

#include <algorithm>
#include <queue>
#include <string>

#include "grpcpp/impl/codegen/call_op_set.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "reverb/cc/client.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/reverb_service_mock.grpc.pb.h"
#include "reverb/cc/support/grpc_util.h"
#include "reverb/cc/support/queue.h"
#include "reverb/cc/support/uint128.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace deepmind {
namespace reverb {
namespace {

using ::deepmind::reverb::testing::Partially;
using ::tensorflow::errors::Internal;
using ::tensorflow::errors::Unavailable;
using ::testing::ElementsAre;
using ::testing::SizeIs;

constexpr auto kNotificationTimeout = absl::Milliseconds(200);

std::vector<tensorflow::Tensor> MakeTimestep(int num_tensors = 1) {
  tensorflow::Tensor tensor(1.0f);
  std::vector<tensorflow::Tensor> res(num_tensors, tensor);
  return res;
}

tensorflow::StructuredValue MakeSignature(
    tensorflow::DataType dtype = tensorflow::DT_FLOAT,
    const tensorflow::PartialTensorShape& shape =
        tensorflow::PartialTensorShape{}) {
  tensorflow::StructuredValue signature;
  auto* spec = signature.mutable_tensor_spec_value();
  spec->set_dtype(dtype);
  spec->set_name("tensor0");
  shape.AsProto(spec->mutable_shape());
  return signature;
}

tensorflow::StructuredValue MakeBoundedTensorSpecSignature(
    tensorflow::DataType dtype = tensorflow::DT_FLOAT,
    const tensorflow::PartialTensorShape& shape =
        tensorflow::PartialTensorShape{},
    const tensorflow::Tensor& min = tensorflow::Tensor(0.0f),
    const tensorflow::Tensor& max = tensorflow::Tensor(10.0f)) {
  tensorflow::StructuredValue signature;
  auto* spec = signature.mutable_bounded_tensor_spec_value();
  spec->set_dtype(dtype);
  spec->set_name("tensor0");
  shape.AsProto(spec->mutable_shape());
  min.AsProtoField(spec->mutable_minimum());
  max.AsProtoField(spec->mutable_maximum());
  return signature;
}

MATCHER(IsChunk, "") { return arg.has_chunk(); }

MATCHER_P4(IsItemWithRangeAndPriorityAndTable, offset, length, priority, table,
           "") {
  return arg.has_item() &&
         arg.item().item().sequence_range().offset() == offset &&
         arg.item().item().sequence_range().length() == length &&
         arg.item().item().priority() == priority &&
         arg.item().item().table() == table;
}

class FakeInsertStream
    : public grpc::ClientReaderWriterInterface<InsertStreamRequest,
                                               InsertStreamResponse> {
 public:
  FakeInsertStream(std::vector<InsertStreamRequest>* requests,
                   int num_success_writes, grpc::Status bad_status,
                   internal::Queue<uint64_t>* response_ids = nullptr)
      : requests_(requests),
        num_success_writes_(num_success_writes),
        bad_status_(std::move(bad_status)),
        response_ids_(response_ids) {}

  bool Write(const InsertStreamRequest& msg,
             grpc::WriteOptions options) override {
    requests_->push_back(msg);
    if (msg.item().send_confirmation()) {
      written_item_ids_.push(msg.item().item().key());
    }
    return num_success_writes_-- > 0;
  }

  bool Read(InsertStreamResponse* response) override {
    // If an explicit response IDs queue was provided then we block until it is
    // nonempty.
    if (response_ids_ != nullptr) {
      uint64_t id;
      CHECK(response_ids_->Pop(&id));
      response->set_key(id);
      return true;
    }

    // If the response IDs queue wasn't explicitly provided then we fallback to
    // return IDs of that has been sent to the fake server
    if (written_item_ids_.empty()) {
      return false;
    }
    response->set_key(written_item_ids_.front());
    written_item_ids_.pop();
    return true;
  }

  grpc::Status Finish() override {
    return num_success_writes_ >= 0 ? grpc::Status::OK : bad_status_;
  }

  bool WritesDone() override { return num_success_writes_-- > 0; }

  bool NextMessageSize(uint32_t* sz) override {
    if (written_item_ids_.empty()) {
      return false;
    }
    InsertStreamResponse response;
    *sz = response.ByteSizeLong();
    return true;
  }

  void WaitForInitialMetadata() override {}

 private:
  std::vector<InsertStreamRequest>* requests_;
  std::queue<uint64_t> written_item_ids_;
  int num_success_writes_;
  grpc::Status bad_status_;
  internal::Queue<uint64_t>* response_ids_;
};

class FakeStub : public /* grpc_gen:: */MockReverbServiceStub {
 public:
  explicit FakeStub(std::list<FakeInsertStream*> streams,
                    const tensorflow::StructuredValue* signature = nullptr)
      : streams_(std::move(streams)) {
    if (signature) {
      *response_.mutable_tables_state_id() =
          Uint128ToMessage(absl::MakeUint128(1, 2));
      auto* table_info = response_.add_table_info();
      table_info->set_name("dist");
      *table_info->mutable_signature() = *signature;
    }
  }
  ~FakeStub() override {
    // Since writers where allocated with New we manually free the memory if
    // the writer hasn't already been passed to the Writer where it is
    // handled as a unique ptr and thus is destroyed with ~Writer.
    while (!streams_.empty()) {
      auto writer = streams_.front();
      delete writer;
      streams_.pop_front();
    }
  }

  grpc::ClientReaderWriterInterface<InsertStreamRequest, InsertStreamResponse>*
  InsertStreamRaw(grpc::ClientContext* context) override {
    auto writer = streams_.front();
    streams_.pop_front();
    return writer;
  }

  grpc::Status ServerInfo(grpc::ClientContext* context,
                          const ServerInfoRequest& request,
                          ServerInfoResponse* response) override {
    *response = response_;
    return grpc::Status::OK;
  }

 private:
  ServerInfoResponse response_;
  std::list<FakeInsertStream*> streams_;
};

std::shared_ptr<FakeStub> MakeGoodStub(
    std::vector<InsertStreamRequest>* requests,
    const tensorflow::StructuredValue* signature = nullptr) {
  FakeInsertStream* stream =
      new FakeInsertStream(requests, 10000, ToGrpcStatus(Internal("")));
  return std::make_shared<FakeStub>(std::list<FakeInsertStream*>{stream},
                                    signature);
}

std::shared_ptr<FakeStub> MakeFlakyStub(
    std::vector<InsertStreamRequest>* requests, int num_success, int num_fail,
    grpc::Status error) {
  std::list<FakeInsertStream*> streams;
  streams.push_back(new FakeInsertStream(requests, num_success, error));
  for (int i = 1; i < num_fail; i++) {
    streams.push_back(new FakeInsertStream(requests, 0, error));
  }
  streams.push_back(
      new FakeInsertStream(requests, 10000, ToGrpcStatus(Internal(""))));
  return std::make_shared<FakeStub>(std::move(streams));
}

TEST(WriterTest, DoesNotSendTimestepsWhenThereAreNoItems) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, 2, 10);
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  EXPECT_THAT(requests, SizeIs(0));
}

TEST(WriterTest, OnlySendsChunksWhichAreUsedByItems) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, 2, 10);
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  EXPECT_THAT(requests, SizeIs(0));

  TF_ASSERT_OK(writer.CreateItem("dist", 3, 1.0));
  ASSERT_THAT(requests, SizeIs(3));
  EXPECT_THAT(requests[0], IsChunk());
  EXPECT_THAT(requests[1], IsChunk());
  EXPECT_THAT(requests[2],
              IsItemWithRangeAndPriorityAndTable(1, 3, 1.0, "dist"));
  EXPECT_THAT(requests[2].item().item().chunk_keys(),
              ElementsAre(requests[0].chunk().chunk_key(),
                          requests[1].chunk().chunk_key()));
}

TEST(WriterTest, DoesNotSendAlreadySentChunks) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, 2, 10);

  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.CreateItem("dist", 1, 1.5));

  ASSERT_THAT(requests, SizeIs(2));

  EXPECT_THAT(requests[0], IsChunk());
  auto first_chunk_key = requests[0].chunk().chunk_key();

  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(1, 1, 1.5, "dist"));
  EXPECT_THAT(requests[1].item().item().chunk_keys(),
              ElementsAre(first_chunk_key));

  requests.clear();
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.CreateItem("dist", 3, 1.3));

  ASSERT_THAT(requests, SizeIs(2));
  EXPECT_THAT(requests[0], IsChunk());
  auto second_chunk_key = requests[0].chunk().chunk_key();

  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(1, 3, 1.3, "dist"));
  EXPECT_THAT(requests[1].item().item().chunk_keys(),
              ElementsAre(first_chunk_key, second_chunk_key));
}

TEST(WriterTest, SendsPendingDataOnClose) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, 2, 10);

  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(0));

  TF_ASSERT_OK(writer.Close());
  ASSERT_THAT(requests, SizeIs(2));
  EXPECT_THAT(requests[0], IsChunk());
  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(0, 1, 1.0, "dist"));
  EXPECT_THAT(requests[1].item().item().chunk_keys(),
              ElementsAre(requests[0].chunk().chunk_key()));
}

TEST(WriterTest, FailsIfMethodsCalledAfterClose) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, 2, 10);

  TF_ASSERT_OK(writer.Close());

  EXPECT_FALSE(writer.Close().ok());
  EXPECT_FALSE(writer.Append(MakeTimestep()).ok());
  EXPECT_FALSE(writer.CreateItem("dist", 1, 1.0).ok());
}

TEST(WriterTest, RetriesOnTransientError) {
  std::vector<InsertStreamRequest> requests;
  // 1 fail, then all success.
  auto stub = MakeFlakyStub(&requests, 0, 1, ToGrpcStatus(Unavailable("")));
  Writer writer(stub, 2, 10);

  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));

  ASSERT_THAT(requests, SizeIs(3));
  EXPECT_THAT(requests[0], IsChunk());
  EXPECT_THAT(requests[1], IsChunk());
  EXPECT_THAT(requests[0], testing::EqualsProto(requests[1]));
  EXPECT_THAT(requests[2],
              IsItemWithRangeAndPriorityAndTable(1, 1, 1.0, "dist"));
  EXPECT_THAT(requests[2].item().item().chunk_keys(),
              ElementsAre(requests[0].chunk().chunk_key()));
}

TEST(WriterTest, DoesNotRetryOnNonTransientError) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeFlakyStub(&requests, 0, 1, ToGrpcStatus(Internal("")));
  Writer writer(stub, 2, 10);

  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  EXPECT_FALSE(writer.CreateItem("dist", 1, 1.0).ok());

  EXPECT_THAT(requests, SizeIs(1));  // Tries only once and then gives up.
}

TEST(WriterTest, CallsCloseWhenObjectDestroyed) {
  std::vector<InsertStreamRequest> requests;
  {
    auto stub = MakeGoodStub(&requests);
    Writer writer(stub, 2, 10);
    TF_ASSERT_OK(writer.Append(MakeTimestep()));
    TF_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));
    EXPECT_THAT(requests, SizeIs(0));
  }
  ASSERT_THAT(requests, SizeIs(2));
}

TEST(WriterTest, ResendsOnlyTheChunksTheRemainingItemsNeedWithNewStream) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeFlakyStub(&requests, 3, 1, ToGrpcStatus(Unavailable("")));
  Writer writer(stub, 2, 10);

  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.CreateItem("dist", 3, 1.0));
  TF_ASSERT_OK(writer.CreateItem("dist2", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(0));

  TF_ASSERT_OK(writer.Append(MakeTimestep()));

  ASSERT_THAT(requests, SizeIs(6));
  EXPECT_THAT(requests[0], IsChunk());
  EXPECT_THAT(requests[1], IsChunk());
  auto first_chunk_key = requests[0].chunk().chunk_key();
  auto second_chunk_key = requests[1].chunk().chunk_key();

  EXPECT_THAT(requests[2],
              IsItemWithRangeAndPriorityAndTable(0, 3, 1.0, "dist"));
  EXPECT_THAT(requests[2].item().item().chunk_keys(),
              ElementsAre(first_chunk_key, second_chunk_key));

  EXPECT_THAT(requests[3], IsItemWithRangeAndPriorityAndTable(
                               0, 1, 1.0, "dist2"));  // Failed
  EXPECT_THAT(requests[3].item().item().chunk_keys(),
              ElementsAre(second_chunk_key));

  // Stream is opened and only the second chunk is sent again.
  EXPECT_THAT(requests[4], IsChunk());
  EXPECT_THAT(requests[5],
              IsItemWithRangeAndPriorityAndTable(0, 1, 1.0, "dist2"));
  EXPECT_THAT(requests[5].item().item().chunk_keys(),
              ElementsAre(second_chunk_key));
}

TEST(WriterTest, TellsServerToKeepStreamedItemsStillInClient) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, 2, 6);

  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));

  ASSERT_THAT(requests, SizeIs(2));
  EXPECT_THAT(requests[0], IsChunk());
  auto first_chunk_key = requests[0].chunk().chunk_key();

  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(1, 1, 1.0, "dist"));
  EXPECT_THAT(requests[1].item().keep_chunk_keys(),
              ElementsAre(first_chunk_key));

  requests.clear();

  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));

  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));

  ASSERT_THAT(requests, SizeIs(2));
  EXPECT_THAT(requests[0], IsChunk());
  auto third_chunk_key = requests[0].chunk().chunk_key();

  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(1, 1, 1.0, "dist"));
  EXPECT_THAT(requests[1].item().keep_chunk_keys(),
              ElementsAre(first_chunk_key, third_chunk_key));

  requests.clear();

  // Now the first chunk will go out of scope
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));

  ASSERT_THAT(requests, SizeIs(2));
  EXPECT_THAT(requests[0], IsChunk());
  auto forth_chunk_key = requests[0].chunk().chunk_key();

  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(1, 1, 1.0, "dist"));
  EXPECT_THAT(requests[1].item().keep_chunk_keys(),
              ElementsAre(third_chunk_key, forth_chunk_key));
}

TEST(WriterTest, IgnoresCloseErrorsIfAllItemsWritten) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeFlakyStub(&requests, /*num_success=*/2,
                            /*num_fail=*/1, ToGrpcStatus(Internal("")));
  Writer writer(stub, /*chunk_length=*/1, /*max_timesteps=*/2);

  // Insert an item and make sure it is flushed to the server.
  TF_EXPECT_OK(writer.Append(MakeTimestep()));
  TF_EXPECT_OK(writer.CreateItem("dist", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(2));

  // Close the writer without any pending items and check that it swallows
  // the error.
  TF_EXPECT_OK(writer.Close());
}

TEST(WriterTest, ReturnsCloseErrorsIfAllItemsNotWritten) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeFlakyStub(&requests, /*num_success=*/1,
                            /*num_fail=*/1, ToGrpcStatus(Internal("")));
  Writer writer(stub, /*chunk_length=*/2, /*max_timesteps=*/4);

  // Insert an item which is shorter
  // than the batch and thus should not
  // be automatically flushed.
  TF_EXPECT_OK(writer.Append(MakeTimestep()));
  TF_EXPECT_OK(writer.CreateItem("dist", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(0));

  // Since not all items where sent
  // before the error should be
  // returned.
  EXPECT_EQ(writer.Close().code(), tensorflow::error::INTERNAL);
}

TEST(WriterTest, FlushWritesItem) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, /*chunk_length=*/2, /*max_timesteps=*/4);

  // Insert an item which is shorter than the batch and thus should not be
  // automatically flushed.
  TF_EXPECT_OK(writer.Append(MakeTimestep()));
  EXPECT_THAT(requests, SizeIs(0));
  // No Item, Flush does nothing.
  TF_EXPECT_OK(writer.Flush());
  EXPECT_THAT(requests, SizeIs(0));
  TF_EXPECT_OK(writer.CreateItem("dist", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(0));
  // Flush the item and make sure it doesn't result in an error.
  TF_EXPECT_OK(writer.Flush());
  EXPECT_THAT(requests, SizeIs(2));
  EXPECT_THAT(requests[0], IsChunk());
  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(0, 1, 1.0, "dist"));

  // Repeat.
  TF_EXPECT_OK(writer.Append(MakeTimestep()));
  EXPECT_THAT(requests, SizeIs(2));
  TF_EXPECT_OK(writer.CreateItem("dist", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(2));
  TF_EXPECT_OK(writer.Flush());
  EXPECT_THAT(requests, SizeIs(4));
  EXPECT_THAT(requests[2], IsChunk());
  EXPECT_THAT(requests[3],
              IsItemWithRangeAndPriorityAndTable(0, 1, 1.0, "dist"));
}

TEST(WriterTest, SequenceRangeIsSetOnChunks) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, /*chunk_length=*/2,
                /*max_timesteps=*/4);

  TF_EXPECT_OK(writer.Append(MakeTimestep()));
  TF_EXPECT_OK(writer.Append(MakeTimestep()));
  TF_EXPECT_OK(writer.Append(MakeTimestep()));
  TF_EXPECT_OK(writer.CreateItem("dist", 3, 1.0));
  TF_EXPECT_OK(writer.Append(MakeTimestep()));

  EXPECT_THAT(
      requests,
      ElementsAre(
          Partially(testing::EqualsProto("chunk: { sequence_range: { start: 0 "
                                         "end: 1 } delta_encoded: false }")),
          Partially(testing::EqualsProto("chunk: { sequence_range: { start: 2 "
                                         "end: 3 } delta_encoded: false }")),
          IsItemWithRangeAndPriorityAndTable(0, 3, 1.0, "dist")));

  EXPECT_NE(requests[0].chunk().sequence_range().episode_id(), 0);
  EXPECT_EQ(requests[0].chunk().sequence_range().episode_id(),
            requests[1].chunk().sequence_range().episode_id());
}

TEST(WriterTest, DeltaEncode) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, /*chunk_length=*/2,
                /*max_timesteps=*/4, /*delta_encoded=*/true);

  TF_EXPECT_OK(writer.Append(MakeTimestep()));
  TF_EXPECT_OK(writer.Append(MakeTimestep()));
  TF_EXPECT_OK(writer.Append(MakeTimestep()));
  TF_EXPECT_OK(writer.CreateItem("dist", 3, 1.0));
  TF_EXPECT_OK(writer.Append(MakeTimestep()));

  EXPECT_THAT(
      requests,
      ElementsAre(
          Partially(testing::EqualsProto("chunk: { sequence_range: { start: 0 "
                                         "end: 1 } delta_encoded: true }")),
          Partially(testing::EqualsProto("chunk: { sequence_range: { start: 2 "
                                         "end: 3 } delta_encoded: true }")),
          IsItemWithRangeAndPriorityAndTable(0, 3, 1.0, "dist")));
}

TEST(WriterTest, MultiChunkItemsAreCorrect) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  Writer writer(stub, /*chunk_length=*/3,
                /*max_timesteps=*/4, /*delta_encoded=*/false);

  // We create two chunks with 5 time steps (t_0,.., t_4) and 3 sequences
  // (s_0, s_1, s_2):
  // +--- CHUNK0 --+- CHUNK1 -+
  // | t_0 t_1 t_2 | t_3 t_4  |
  // +-------------+----------+
  // | s_0 s_0 s_1 | s_1 s_3  |
  // +-------------+----------+

  // First item: 1 chunk.
  TF_EXPECT_OK(writer.Append(MakeTimestep()));
  TF_EXPECT_OK(writer.Append(MakeTimestep()));
  TF_EXPECT_OK(writer.CreateItem("dist", 2, 1.0));

  // Second item: 2 chunks.
  TF_EXPECT_OK(writer.Append(MakeTimestep()));
  TF_EXPECT_OK(writer.Append(MakeTimestep()));
  TF_EXPECT_OK(writer.CreateItem("dist", 2, 1.0));

  // Third item: 1 chunk.
  TF_EXPECT_OK(writer.Append(MakeTimestep()));
  TF_EXPECT_OK(writer.CreateItem("dist", 1, 1.0));

  TF_EXPECT_OK(writer.Close());

  EXPECT_THAT(
      requests,
      ElementsAre(
          Partially(testing::EqualsProto("chunk: { sequence_range: { start: 0 "
                                         "end: 2 } delta_encoded: false }")),
          IsItemWithRangeAndPriorityAndTable(0, 2, 1.0, "dist"),
          Partially(testing::EqualsProto("chunk: { sequence_range: { start: 3 "
                                         "end: 4 } delta_encoded: false }")),
          IsItemWithRangeAndPriorityAndTable(2, 2, 1.0, "dist"),
          IsItemWithRangeAndPriorityAndTable(1, 1, 1.0, "dist")));

  EXPECT_EQ(requests[1].item().item().chunk_keys_size(), 1);
  EXPECT_EQ(requests[3].item().item().chunk_keys_size(), 2);
  EXPECT_EQ(requests[4].item().item().chunk_keys_size(), 1);
}

TEST(WriterTest, WriteTimeStepsMatchingSignature) {
  std::vector<InsertStreamRequest> requests;
  tensorflow::StructuredValue signature =
      MakeSignature(tensorflow::DT_FLOAT, tensorflow::PartialTensorShape({}));
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  TF_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  TF_ASSERT_OK(writer->Append(MakeTimestep()));
  TF_ASSERT_OK(writer->Append(MakeTimestep()));
  TF_ASSERT_OK(writer->CreateItem("dist", 2, 1.0));
  ASSERT_THAT(requests, SizeIs(2));
}

TEST(WriterTest, WriteTimeStepsMatchingBoundedSignature) {
  std::vector<InsertStreamRequest> requests;
  tensorflow::StructuredValue signature = MakeBoundedTensorSpecSignature(
      tensorflow::DT_FLOAT, tensorflow::PartialTensorShape({}));
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  TF_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  TF_ASSERT_OK(writer->Append(MakeTimestep()));
  TF_ASSERT_OK(writer->Append(MakeTimestep()));
  TF_ASSERT_OK(writer->CreateItem("dist", 2, 1.0));
  ASSERT_THAT(requests, SizeIs(2));
}

TEST(WriterTest, WriteTimeStepsNumTensorsDontMatchSignatureError) {
  std::vector<InsertStreamRequest> requests;
  tensorflow::StructuredValue signature = MakeSignature();
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  TF_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  TF_ASSERT_OK(writer->Append(MakeTimestep(/*num_tensors=*/2)));
  TF_ASSERT_OK(writer->Append(MakeTimestep(/*num_tensors=*/2)));
  auto status = writer->CreateItem("dist", 2, 1.0);
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(
      status.error_message(),
      ::testing::HasSubstr(
          "Append for timestep offset 0 was called with 2 tensors, "
          "but table requires 1 tensors per entry."));
}

TEST(WriterTest, WriteTimeStepsNumTensorsDontMatchBoundedSignatureError) {
  std::vector<InsertStreamRequest> requests;
  tensorflow::StructuredValue signature = MakeBoundedTensorSpecSignature();
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  TF_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  TF_ASSERT_OK(writer->Append(MakeTimestep(/*num_tensors=*/2)));
  TF_ASSERT_OK(writer->Append(MakeTimestep(/*num_tensors=*/2)));
  auto status = writer->CreateItem("dist", 2, 1.0);
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr(
                  "Append for timestep offset 0 was called with 2 tensors, "
                  "but table requires 1 tensors per entry."));
}

TEST(WriterTest, WriteTimeStepsInconsistentDtypeError) {
  std::vector<InsertStreamRequest> requests;
  tensorflow::StructuredValue signature = MakeSignature(tensorflow::DT_INT32);
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  TF_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  TF_ASSERT_OK(writer->Append(MakeTimestep()));
  TF_ASSERT_OK(writer->Append(MakeTimestep()));
  auto status = writer->CreateItem("dist", 2, 1.0);
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr(
                  "timestep offset 0 in (flattened) tensor location 0 with "
                  "dtype float and shape [] but expected a tensor of dtype "
                  "int32 and shape compatible with <unknown>"));
}

TEST(WriterTest, WriteTimeStepsInconsistentDtypeErrorAgainstBoundedSpec) {
  std::vector<InsertStreamRequest> requests;
  tensorflow::StructuredValue signature =
      MakeBoundedTensorSpecSignature(tensorflow::DT_INT32);
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  TF_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  TF_ASSERT_OK(writer->Append(MakeTimestep()));
  TF_ASSERT_OK(writer->Append(MakeTimestep()));
  auto status = writer->CreateItem("dist", 2, 1.0);
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr(
                  "timestep offset 0 in (flattened) tensor location 0 with "
                  "dtype float and shape [] but expected a tensor of dtype "
                  "int32 and shape compatible with <unknown>"));
}

TEST(WriterTest, WriteTimeStepsInconsistentShapeError) {
  std::vector<InsertStreamRequest> requests;
  tensorflow::StructuredValue signature =
      MakeSignature(tensorflow::DT_FLOAT, tensorflow::PartialTensorShape({-1}));
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  TF_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  TF_ASSERT_OK(writer->Append(MakeTimestep()));
  TF_ASSERT_OK(writer->Append(MakeTimestep()));
  auto status = writer->CreateItem("dist", 2, 1.0);
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr(
                  "timestep offset 0 in (flattened) tensor location 0 with "
                  "dtype float and shape [] but expected a tensor of dtype "
                  "float and shape compatible with [?]"));
}

TEST(WriterTest, WriteTimeStepsInconsistentShapeErrorAgainstBoundedSpec) {
  std::vector<InsertStreamRequest> requests;
  tensorflow::StructuredValue signature = MakeBoundedTensorSpecSignature(
      tensorflow::DT_FLOAT, tensorflow::PartialTensorShape({-1}));
  auto stub = MakeGoodStub(&requests, &signature);
  Client client(stub);
  std::unique_ptr<Writer> writer;
  TF_EXPECT_OK(client.NewWriter(2, 6, /*delta_encoded=*/false, &writer));

  TF_ASSERT_OK(writer->Append(MakeTimestep()));
  TF_ASSERT_OK(writer->Append(MakeTimestep()));
  auto status = writer->CreateItem("dist", 2, 1.0);
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr(
                  "timestep offset 0 in (flattened) tensor location 0 with "
                  "dtype float and shape [] but expected a tensor of dtype "
                  "float and shape compatible with [?]"));
}

std::pair<std::shared_ptr<FakeStub>, std::unique_ptr<internal::Queue<uint64_t>>>
MakeStubWithExplicitResponseQueue(std::vector<InsertStreamRequest>* requests) {
  auto response_ids = absl::make_unique<internal::Queue<uint64_t>>(100);
  FakeInsertStream* stream = new FakeInsertStream(
      requests, 10000, ToGrpcStatus(Internal("")), response_ids.get());
  auto stub = std::make_shared<FakeStub>(std::list<FakeInsertStream*>{stream});
  return {std::move(stub), std::move(response_ids)};
}

TEST(WriterTest, CloseBlocksUntilAllItemsConfirmed) {
  std::vector<InsertStreamRequest> requests;
  auto pair = MakeStubWithExplicitResponseQueue(&requests);
  auto response_ids = std::move(pair.second);
  Writer writer(pair.first, 2, 2, false, nullptr, 100);

  // Creating the items should not result in any blocking as the number of in
  // flight items is 100.
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.CreateItem("dist", 2, 1.0));

  // Attempt to close the writer from a separate thread.
  absl::Notification notification;
  auto close_thread = internal::StartThread("Close", [&writer, &notification] {
    REVERB_CHECK(writer.Close().ok());
    notification.Notify();
  });

  // The close call should not be able to complete as the server hasn't
  // confirmed that all the IDs have been written.
  EXPECT_FALSE(
      notification.WaitForNotificationWithTimeout(kNotificationTimeout));

  // Sending the response for the first item should not be enough.
  ASSERT_TRUE(response_ids->Push(1));
  EXPECT_FALSE(
      notification.WaitForNotificationWithTimeout(kNotificationTimeout));

  // Sending response for the second (and last) item should unblock the Close
  // call.
  ASSERT_TRUE(response_ids->Push(2));
  notification.WaitForNotification();
}

TEST(WriterTest, FlushBlocksUntilAllItemsConfirmed) {
  std::vector<InsertStreamRequest> requests;
  auto pair = MakeStubWithExplicitResponseQueue(&requests);
  auto response_ids = std::move(pair.second);
  Writer writer(pair.first, 2, 2, false, nullptr, 100);

  // Creating the items should not result in any blocking as the number of in
  // flight items is 100.
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.CreateItem("dist", 2, 1.0));

  // Attempt to flush the writer from a separate thread.
  absl::Notification notification;
  auto flush_thread = internal::StartThread("Flush", [&writer, &notification] {
    REVERB_CHECK(writer.Flush().ok());
    notification.Notify();
  });

  // The flush call should not be able to complete as the server hasn't
  // confirmed that all the IDs have been written.
  EXPECT_FALSE(
      notification.WaitForNotificationWithTimeout(kNotificationTimeout));

  // Sending the response for the first item should not be enough.
  ASSERT_TRUE(response_ids->Push(1));
  EXPECT_FALSE(
      notification.WaitForNotificationWithTimeout(kNotificationTimeout));

  // Sending response for the second (and last) item should unblock the Close
  // call.
  ASSERT_TRUE(response_ids->Push(2));
  notification.WaitForNotification();
}

TEST(WriterTest, BlocksWhenMaxInFlighItemsReached) {
  std::vector<InsertStreamRequest> requests;
  auto pair = MakeStubWithExplicitResponseQueue(&requests);
  auto response_ids = std::move(pair.second);
  Writer writer(pair.first, 1, 2, false, nullptr, 2);

  // Creating two items should not result in any blocking as it does not exceed
  // the maximum number of in flight items.
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.CreateItem("dist", 1, 1.0));
  TF_ASSERT_OK(writer.Append(MakeTimestep()));
  TF_ASSERT_OK(writer.CreateItem("dist", 2, 1.0));

  // Attempt to send one more items in a separate thread.
  absl::Notification notification;
  auto thread = internal::StartThread("InsertItem", [&writer, &notification] {
    TF_ASSERT_OK(writer.Append(MakeTimestep()));
    TF_ASSERT_OK(writer.CreateItem("dist", 2, 1.0));
    notification.Notify();
  });

  // Initially the third item should be blocked.
  ASSERT_FALSE(
      notification.WaitForNotificationWithTimeout(kNotificationTimeout));

  // Confirming one item should unblock the pending item.
  ASSERT_TRUE(response_ids->Push(1));
  notification.WaitForNotification();

  // Send the remaining responses to unblock any pending reads without
  // cancelling the stream.
  ASSERT_TRUE(response_ids->Push(2));
  ASSERT_TRUE(response_ids->Push(3));
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
