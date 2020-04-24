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

#include "reverb/cc/replay_writer.h"

#include <algorithm>
#include <string>

#include "grpcpp/impl/codegen/call_op_set.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/impl/codegen/sync_stream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/replay_client.h"
#include "reverb/cc/replay_service.grpc.pb.h"
#include "reverb/cc/replay_service_mock.grpc.pb.h"
#include "reverb/cc/support/grpc_util.h"
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
using ::tensorflow::errors::DeadlineExceeded;
using ::tensorflow::errors::Internal;
using ::tensorflow::errors::Unavailable;
using ::testing::ElementsAre;
using ::testing::SizeIs;

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

MATCHER(IsChunk, "") { return arg.has_chunk(); }

MATCHER_P4(IsItemWithRangeAndPriorityAndTable, offset, length, priority, table,
           "") {
  return arg.has_item() &&
         arg.item().item().sequence_range().offset() == offset &&
         arg.item().item().sequence_range().length() == length &&
         arg.item().item().priority() == priority &&
         arg.item().item().table() == table;
}

class FakeWriter : public grpc::ClientWriterInterface<InsertStreamRequest> {
 public:
  FakeWriter(std::vector<InsertStreamRequest>* requests, int num_success_writes,
             grpc::Status bad_status)
      : requests_(requests),
        num_success_writes_(num_success_writes),
        bad_status_(std::move(bad_status)) {}

  bool Write(const InsertStreamRequest& msg,
             grpc::WriteOptions options) override {
    requests_->push_back(msg);
    return num_success_writes_-- > 0;
  }

  grpc::Status Finish() override {
    return num_success_writes_ >= 0 ? grpc::Status::OK : bad_status_;
  }

  bool WritesDone() override { return num_success_writes_-- > 0; }

 private:
  std::vector<InsertStreamRequest>* requests_;
  int num_success_writes_;
  grpc::Status bad_status_;
};

class FakeStub : public /* grpc_gen:: */MockReplayServiceStub {
 public:
  explicit FakeStub(std::list<FakeWriter*> writers,
                    const tensorflow::StructuredValue* signature = nullptr)
      : writers_(std::move(writers)) {
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
    // the writer hasn't already been passed to the ReplayWriter where it is
    // handled as a unique ptr and thus is destroyed with ~ReplayWriter.
    while (!writers_.empty()) {
      auto writer = writers_.front();
      delete writer;
      writers_.pop_front();
    }
  }

  grpc::ClientWriterInterface<InsertStreamRequest>* InsertStreamRaw(
      grpc::ClientContext* context, InsertStreamResponse* response) override {
    auto writer = writers_.front();
    writers_.pop_front();
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
  std::list<FakeWriter*> writers_;
};

std::shared_ptr<FakeStub> MakeGoodStub(
    std::vector<InsertStreamRequest>* requests,
    const tensorflow::StructuredValue* signature = nullptr) {
  FakeWriter* writer =
      new FakeWriter(requests, 10000, ToGrpcStatus(Internal("")));
  return std::make_shared<FakeStub>(std::list<FakeWriter*>{writer}, signature);
}

std::shared_ptr<FakeStub> MakeFlakyStub(
    std::vector<InsertStreamRequest>* requests, int num_success, int num_fail,
    grpc::Status error) {
  std::list<FakeWriter*> writers;
  writers.push_back(new FakeWriter(requests, num_success, error));
  for (int i = 1; i < num_fail; i++) {
    writers.push_back(new FakeWriter(requests, 0, error));
  }
  writers.push_back(
      new FakeWriter(requests, 10000, ToGrpcStatus(Internal(""))));
  return std::make_shared<FakeStub>(std::move(writers));
}

TEST(ReplayWriterTest, DoesNotSendTimestepsWhenThereAreNoItems) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  ReplayWriter client(stub, 2, 10);
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  EXPECT_THAT(requests, SizeIs(0));
}

TEST(ReplayWriterTest, OnlySendsChunksWhichAreUsedByItems) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  ReplayWriter client(stub, 2, 10);
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  EXPECT_THAT(requests, SizeIs(0));

  TF_ASSERT_OK(client.AddPriority("dist", 3, 1.0));
  ASSERT_THAT(requests, SizeIs(3));
  EXPECT_THAT(requests[0], IsChunk());
  EXPECT_THAT(requests[1], IsChunk());
  EXPECT_THAT(requests[2],
              IsItemWithRangeAndPriorityAndTable(1, 3, 1.0, "dist"));
  EXPECT_THAT(requests[2].item().item().chunk_keys(),
              ElementsAre(requests[0].chunk().chunk_key(),
                          requests[1].chunk().chunk_key()));
}

TEST(ReplayWriterTest, DoesNotSendAlreadySentChunks) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  ReplayWriter client(stub, 2, 10);

  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AddPriority("dist", 1, 1.5));

  ASSERT_THAT(requests, SizeIs(2));

  EXPECT_THAT(requests[0], IsChunk());
  auto first_chunk_key = requests[0].chunk().chunk_key();

  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(1, 1, 1.5, "dist"));
  EXPECT_THAT(requests[1].item().item().chunk_keys(),
              ElementsAre(first_chunk_key));

  requests.clear();
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AddPriority("dist", 3, 1.3));

  ASSERT_THAT(requests, SizeIs(2));
  EXPECT_THAT(requests[0], IsChunk());
  auto second_chunk_key = requests[0].chunk().chunk_key();

  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(1, 3, 1.3, "dist"));
  EXPECT_THAT(requests[1].item().item().chunk_keys(),
              ElementsAre(first_chunk_key, second_chunk_key));
}

TEST(ReplayWriterTest, SendsPendingDataOnClose) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  ReplayWriter client(stub, 2, 10);

  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AddPriority("dist", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(0));

  TF_ASSERT_OK(client.Close());
  ASSERT_THAT(requests, SizeIs(2));
  EXPECT_THAT(requests[0], IsChunk());
  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(0, 1, 1.0, "dist"));
  EXPECT_THAT(requests[1].item().item().chunk_keys(),
              ElementsAre(requests[0].chunk().chunk_key()));
}

TEST(ReplayWriterTest, FailsIfMethodsCalledAfterClose) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  ReplayWriter client(stub, 2, 10);

  TF_ASSERT_OK(client.Close());

  EXPECT_FALSE(client.Close().ok());
  EXPECT_FALSE(client.AppendTimestep(MakeTimestep()).ok());
  EXPECT_FALSE(client.AddPriority("dist", 1, 1.0).ok());
}

TEST(ReplayWriterTest, RetriesOnTransientError) {
  std::vector<tensorflow::Status> transient_errors(
      {DeadlineExceeded(""), Unavailable("")});

  for (const auto& error : transient_errors) {
    std::vector<InsertStreamRequest> requests;
    // 1 fail, then all success.
    auto stub = MakeFlakyStub(&requests, 0, 1, ToGrpcStatus(error));
    ReplayWriter client(stub, 2, 10);

    TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
    TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
    TF_ASSERT_OK(client.AddPriority("dist", 1, 1.0));

    ASSERT_THAT(requests, SizeIs(3));
    EXPECT_THAT(requests[0], IsChunk());
    EXPECT_THAT(requests[1], IsChunk());
    EXPECT_THAT(requests[0], testing::EqualsProto(requests[1]));
    EXPECT_THAT(requests[2],
                IsItemWithRangeAndPriorityAndTable(1, 1, 1.0, "dist"));
    EXPECT_THAT(requests[2].item().item().chunk_keys(),
                ElementsAre(requests[0].chunk().chunk_key()));
  }
}

TEST(ReplayWriterTest, DoesNotRetryOnNonTransientError) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeFlakyStub(&requests, 0, 1, ToGrpcStatus(Internal("")));
  ReplayWriter client(stub, 2, 10);

  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  EXPECT_FALSE(client.AddPriority("dist", 1, 1.0).ok());

  EXPECT_THAT(requests, SizeIs(1));  // Tries only once and then gives up.
}

TEST(ReplayWriterTest, CallsCloseWhenObjectDestroyed) {
  std::vector<InsertStreamRequest> requests;
  {
    auto stub = MakeGoodStub(&requests);
    ReplayWriter client(stub, 2, 10);
    TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
    TF_ASSERT_OK(client.AddPriority("dist", 1, 1.0));
    EXPECT_THAT(requests, SizeIs(0));
  }
  ASSERT_THAT(requests, SizeIs(2));
}

TEST(ReplayWriterTest, ResendsOnlyTheChunksTheRemainingItemsNeedWithNewStream) {
  std::vector<InsertStreamRequest> requests;
  auto stub =
      MakeFlakyStub(&requests, 3, 1, ToGrpcStatus(DeadlineExceeded("")));
  ReplayWriter client(stub, 2, 10);

  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AddPriority("dist", 3, 1.0));
  TF_ASSERT_OK(client.AddPriority("dist2", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(0));

  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));

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

TEST(ReplayWriterTest, TellsServerToKeepStreamedItemsStillInClient) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  ReplayWriter client(stub, 2, 6);

  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AddPriority("dist", 1, 1.0));

  ASSERT_THAT(requests, SizeIs(2));
  EXPECT_THAT(requests[0], IsChunk());
  auto first_chunk_key = requests[0].chunk().chunk_key();

  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(1, 1, 1.0, "dist"));
  EXPECT_THAT(requests[1].item().keep_chunk_keys(),
              ElementsAre(first_chunk_key));

  requests.clear();

  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));

  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AddPriority("dist", 1, 1.0));

  ASSERT_THAT(requests, SizeIs(2));
  EXPECT_THAT(requests[0], IsChunk());
  auto third_chunk_key = requests[0].chunk().chunk_key();

  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(1, 1, 1.0, "dist"));
  EXPECT_THAT(requests[1].item().keep_chunk_keys(),
              ElementsAre(first_chunk_key, third_chunk_key));

  requests.clear();

  // Now the first chunk will go out of scope
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client.AddPriority("dist", 1, 1.0));

  ASSERT_THAT(requests, SizeIs(2));
  EXPECT_THAT(requests[0], IsChunk());
  auto forth_chunk_key = requests[0].chunk().chunk_key();

  EXPECT_THAT(requests[1],
              IsItemWithRangeAndPriorityAndTable(1, 1, 1.0, "dist"));
  EXPECT_THAT(requests[1].item().keep_chunk_keys(),
              ElementsAre(third_chunk_key, forth_chunk_key));
}

TEST(ReplayWriterTest, IgnoresCloseErrorsIfAllItemsWritten) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeFlakyStub(&requests, /*num_success=*/2,
                            /*num_fail=*/1, ToGrpcStatus(Internal("")));
  ReplayWriter client(stub, /*chunk_length=*/1, /*max_timesteps=*/2);

  // Insert an item and make sure it is flushed to the server.
  TF_EXPECT_OK(client.AppendTimestep(MakeTimestep()));
  TF_EXPECT_OK(client.AddPriority("dist", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(2));

  // Close the client without any pending items and check that it swallows
  // the error.
  TF_EXPECT_OK(client.Close());
}

TEST(ReplayWriterTest, ReturnsCloseErrorsIfAllItemsNotWritten) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeFlakyStub(&requests, /*num_success=*/1,
                            /*num_fail=*/1, ToGrpcStatus(Internal("")));
  ReplayWriter client(stub, /*chunk_length=*/2, /*max_timesteps=*/4);

  // Insert an item which is shorter
  // than the batch and thus should not
  // be automatically flushed.
  TF_EXPECT_OK(client.AppendTimestep(MakeTimestep()));
  TF_EXPECT_OK(client.AddPriority("dist", 1, 1.0));
  EXPECT_THAT(requests, SizeIs(0));

  // Since not all items where sent
  // before the error should be
  // returned.
  EXPECT_EQ(client.Close().code(), tensorflow::error::INTERNAL);
}

TEST(ReplayWriterTest, SequenceRangeIsSetOnChunks) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  ReplayWriter client(stub, /*chunk_length=*/2,
                      /*max_timesteps=*/4);

  TF_EXPECT_OK(client.AppendTimestep(MakeTimestep()));
  TF_EXPECT_OK(client.AppendTimestep(MakeTimestep()));
  TF_EXPECT_OK(client.AppendTimestep(MakeTimestep()));
  TF_EXPECT_OK(client.AddPriority("dist", 3, 1.0));
  TF_EXPECT_OK(client.AppendTimestep(MakeTimestep()));

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

TEST(ReplayWriterTest, DeltaEncode) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  ReplayWriter client(stub, /*chunk_length=*/2,
                      /*max_timesteps=*/4, /*delta_encoded=*/true);

  TF_EXPECT_OK(client.AppendTimestep(MakeTimestep()));
  TF_EXPECT_OK(client.AppendTimestep(MakeTimestep()));
  TF_EXPECT_OK(client.AppendTimestep(MakeTimestep()));
  TF_EXPECT_OK(client.AddPriority("dist", 3, 1.0));
  TF_EXPECT_OK(client.AppendTimestep(MakeTimestep()));

  EXPECT_THAT(
      requests,
      ElementsAre(
          Partially(testing::EqualsProto("chunk: { sequence_range: { start: 0 "
                                         "end: 1 } delta_encoded: true }")),
          Partially(testing::EqualsProto("chunk: { sequence_range: { start: 2 "
                                         "end: 3 } delta_encoded: true }")),
          IsItemWithRangeAndPriorityAndTable(0, 3, 1.0, "dist")));
}

TEST(ReplayWriterTest, MultiChunkItemsAreCorrect) {
  std::vector<InsertStreamRequest> requests;
  auto stub = MakeGoodStub(&requests);
  ReplayWriter client(stub, /*chunk_length=*/3,
                      /*max_timesteps=*/4, /*delta_encoded=*/false);

  // We create two chunks with 5 time steps (t_0,.., t_4) and 3 sequences
  // (s_0, s_1, s_2):
  // +--- CHUNK0 --+- CHUNK1 -+
  // | t_0 t_1 t_2 | t_3 t_4  |
  // +-------------+----------+
  // | s_0 s_0 s_1 | s_1 s_3  |
  // +-------------+----------+

  // First item: 1 chunk.
  TF_EXPECT_OK(client.AppendTimestep(MakeTimestep()));
  TF_EXPECT_OK(client.AppendTimestep(MakeTimestep()));
  TF_EXPECT_OK(client.AddPriority("dist", 2, 1.0));

  // Second item: 2 chunks.
  TF_EXPECT_OK(client.AppendTimestep(MakeTimestep()));
  TF_EXPECT_OK(client.AppendTimestep(MakeTimestep()));
  TF_EXPECT_OK(client.AddPriority("dist", 2, 1.0));

  // Third item: 1 chunk.
  TF_EXPECT_OK(client.AppendTimestep(MakeTimestep()));
  TF_EXPECT_OK(client.AddPriority("dist", 1, 1.0));

  TF_EXPECT_OK(client.Close());

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

TEST(ReplayWriterTest, WriteTimeStepsMatchingSignature) {
  std::vector<InsertStreamRequest> requests;
  tensorflow::StructuredValue signature =
      MakeSignature(tensorflow::DT_FLOAT, tensorflow::PartialTensorShape({}));
  auto stub = MakeGoodStub(&requests, &signature);
  ReplayClient replay_client(stub);
  std::unique_ptr<ReplayWriter> client;
  TF_EXPECT_OK(replay_client.NewWriter(2, 6, /*delta_encoded=*/false, &client));

  TF_ASSERT_OK(client->AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client->AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client->AddPriority("dist", 2, 1.0));
  ASSERT_THAT(requests, SizeIs(2));
}

TEST(ReplayWriterTest, WriteTimeStepsNumTensorsDontMatchSignatureError) {
  std::vector<InsertStreamRequest> requests;
  tensorflow::StructuredValue signature = MakeSignature();
  auto stub = MakeGoodStub(&requests, &signature);
  ReplayClient replay_client(stub);
  std::unique_ptr<ReplayWriter> client;
  TF_EXPECT_OK(replay_client.NewWriter(2, 6, /*delta_encoded=*/false, &client));

  TF_ASSERT_OK(client->AppendTimestep(MakeTimestep(/*num_tensors=*/2)));
  TF_ASSERT_OK(client->AppendTimestep(MakeTimestep(/*num_tensors=*/2)));
  auto status = client->AddPriority("dist", 2, 1.0);
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(
      status.error_message(),
      ::testing::HasSubstr(
          "AppendTimestep for timestep offset 0 was called with 2 tensors, "
          "but table requires 1 tensors per entry."));
}

TEST(ReplayWriterTest, WriteTimeStepsInconsistentDtypeError) {
  std::vector<InsertStreamRequest> requests;
  tensorflow::StructuredValue signature = MakeSignature(tensorflow::DT_INT32);
  auto stub = MakeGoodStub(&requests, &signature);
  ReplayClient replay_client(stub);
  std::unique_ptr<ReplayWriter> client;
  TF_EXPECT_OK(replay_client.NewWriter(2, 6, /*delta_encoded=*/false, &client));

  TF_ASSERT_OK(client->AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client->AppendTimestep(MakeTimestep()));
  auto status = client->AddPriority("dist", 2, 1.0);
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr(
                  "timestep offset 0 in (flattened) tensor location 0 with "
                  "dtype float and shape [] but expected a tensor of dtype "
                  "int32 and shape compatible with <unknown>"));
}

TEST(ReplayWriterTest, WriteTimeStepsInconsistentShapeError) {
  std::vector<InsertStreamRequest> requests;
  tensorflow::StructuredValue signature =
      MakeSignature(tensorflow::DT_FLOAT, tensorflow::PartialTensorShape({-1}));
  auto stub = MakeGoodStub(&requests, &signature);
  ReplayClient replay_client(stub);
  std::unique_ptr<ReplayWriter> client;
  TF_EXPECT_OK(replay_client.NewWriter(2, 6, /*delta_encoded=*/false, &client));

  TF_ASSERT_OK(client->AppendTimestep(MakeTimestep()));
  TF_ASSERT_OK(client->AppendTimestep(MakeTimestep()));
  auto status = client->AddPriority("dist", 2, 1.0);
  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr(
                  "timestep offset 0 in (flattened) tensor location 0 with "
                  "dtype float and shape [] but expected a tensor of dtype "
                  "float and shape compatible with [?]"));
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
