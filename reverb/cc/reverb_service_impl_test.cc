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

#include "reverb/cc/reverb_service_impl.h"

#include <cfloat>
#include <cstddef>
#include <list>
#include <memory>
#include <queue>
#include <vector>

#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/server_callback.h"
#include "grpcpp/test/default_reactor_test_peer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "reverb/cc/platform/checkpointing.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/selectors/interface.h"
#include "reverb/cc/selectors/uniform.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace deepmind {
namespace reverb {
namespace {

void WaitForTableSize(Table* table, int size) {
  for (int retry = 0; retry < 100 && size != 0; retry++) {
    absl::SleepFor(absl::Milliseconds(1));
  }
  EXPECT_EQ(table->size(), size);
}

class FakeSelector : public ItemSelector {
 public:
  FakeSelector(std::queue<absl::Status> insert_statuses)
      : insert_statuses_(std::move(insert_statuses)) {}

  absl::Status Delete(Key key) override {
    return absl::OkStatus();
  }

  absl::Status Insert(Key key, double priority) override {
    if (insert_statuses_.empty()) {
      return absl::OkStatus();
    }
    auto status = insert_statuses_.front();
    insert_statuses_.pop();
    return status;
  }

  absl::Status Update(Key key, double priority) override {
    return absl::OkStatus();
  }

  KeyWithProbability Sample() override { return {1, 1}; }

  void Clear() override{};

  KeyDistributionOptions options() const override {
    return KeyDistributionOptions();
  }

  std::string DebugString() const override { return "FakeSelector"; }

 private:
  std::queue<absl::Status> insert_statuses_;
};

const int64_t kMinSizeToSample = 1;
const double kSamplesPerInsert = 1.0;
const double kMinDiff = -DBL_MAX;
const double kMaxDiff = DBL_MAX;

int64_t nextId = 1;

InsertStreamRequest InsertMultiChunkRequest(const std::vector<int64_t>& keys) {
  InsertStreamRequest request;
  for (auto key : keys) {
    request.add_chunks()->set_chunk_key(key);
  }
  return request;
}

InsertStreamRequest InsertChunkRequest(int64_t key) {
  return InsertMultiChunkRequest({key});
}

InsertStreamRequest InsertItemRequest(
    absl::string_view table, const std::vector<int64_t>& sequence_chunks,
    const std::vector<int64_t>& keep_chunks = {},
    bool send_confirmation = false) {
  PrioritizedItem item;
  item.set_key(nextId++);
  item.set_table(table.data(), table.size());

  if (!sequence_chunks.empty()) {
    auto* col = item.mutable_flat_trajectory()->add_columns();
    for (auto chunk_key : sequence_chunks) {
      auto* slice = col->add_chunk_slices();
      slice->set_chunk_key(chunk_key);
      slice->set_index(0);
      slice->set_offset(0);
      slice->set_length(100 / sequence_chunks.size());
    }
  }

  InsertStreamRequest request;
  *request.mutable_item()->mutable_keep_chunk_keys() = {keep_chunks.begin(),
                                                        keep_chunks.end()};
  *request.mutable_item()->mutable_item() = item;
  request.mutable_item()->set_send_confirmation(send_confirmation);

  return request;
}

SampleStreamRequest SampleRequest(absl::string_view table, int num_samples,
                                  int flexible_batch_size = -1) {
  SampleStreamRequest request;
  request.set_table(table.data(), table.size());
  request.set_num_samples(num_samples);
  request.set_flexible_batch_size(flexible_batch_size);

  return request;
}

tensorflow::StructuredValue MakeSignature() {
  tensorflow::StructuredValue signature;
  auto* tensor_spec = signature.mutable_tensor_spec_value();
  tensor_spec->set_name("item0");
  tensorflow::TensorShape().AsProto(tensor_spec->mutable_shape());
  tensor_spec->set_dtype(tensorflow::DT_INT32);
  return signature;
}

std::unique_ptr<RateLimiter> MakeLimiter() {
  return absl::make_unique<RateLimiter>(kSamplesPerInsert, kMinSizeToSample,
                                        kMinDiff, kMaxDiff);
}

std::unique_ptr<ReverbServiceImpl> MakeService(
    int max_size, std::unique_ptr<Checkpointer> checkpointer,
    std::vector<std::shared_ptr<Table>> tables = {}) {
  tables.push_back(absl::make_unique<Table>(
      /*name=*/"dist",
      /*sampler=*/absl::make_unique<UniformSelector>(),
      /*remover=*/absl::make_unique<FifoSelector>(),
      /*max_size=*/max_size,
      /*max_times_sampled=*/0,
      /*rate_limiter=*/MakeLimiter(),
      /*extensions=*/std::vector<std::shared_ptr<TableExtension>>(),
      /*signature=*/absl::make_optional(MakeSignature())));
  std::unique_ptr<ReverbServiceImpl> service;
  REVERB_CHECK_OK(ReverbServiceImpl::Create(
      std::move(tables), std::move(checkpointer), &service));
  return service;
}

std::unique_ptr<ReverbServiceImpl> MakeService(int max_size) {
  return MakeService(max_size, nullptr);
}

TEST(ReverbServiceImplTest, InsertSameItemWorks) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));
  grpc::ClientContext context;
  auto insert_stream = stub.InsertStream(&context);
  ASSERT_TRUE(insert_stream->Write(InsertMultiChunkRequest({1, 2})));
  ASSERT_TRUE(insert_stream->Write(InsertChunkRequest(3)));

  InsertStreamRequest insert_request = InsertItemRequest("dist", {2, 3});
  ASSERT_TRUE(insert_stream->Write(insert_request));

  // The same item again.
  ASSERT_TRUE(insert_stream->Write(InsertMultiChunkRequest({1, 2})));
  ASSERT_TRUE(insert_stream->Write(InsertChunkRequest(3)));
  ASSERT_TRUE(insert_stream->Write(insert_request));
  ASSERT_TRUE(insert_stream->WritesDone());
  REVERB_EXPECT_OK(insert_stream->Finish());
  PrioritizedItem item = insert_request.item().item();
}

TEST(ReverbServiceImplTest, SampleAfterInsertWorks) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));
  grpc::ClientContext context;
  auto insert_stream = stub.InsertStream(&context);
  ASSERT_TRUE(insert_stream->Write(InsertMultiChunkRequest({1, 2})));
  ASSERT_TRUE(insert_stream->Write(InsertChunkRequest(3)));

  InsertStreamRequest insert_request = InsertItemRequest("dist", {2, 3});
  ASSERT_TRUE(insert_stream->Write(insert_request));
  ASSERT_TRUE(insert_stream->WritesDone());
  REVERB_EXPECT_OK(insert_stream->Finish());
  PrioritizedItem item = insert_request.item().item();

  for (int i = 0; i < 5; i++) {
    grpc::ClientContext sample_context;
    auto sample_stream = stub.SampleStream(&sample_context);
    SampleStreamRequest sample_request = SampleRequest("dist", 2, 2);
    SampleStreamResponse sample_response;
    SampleStreamResponse sample_response2;
    sample_stream->Write(sample_request);
    sample_stream->WritesDone();
    ASSERT_TRUE(sample_stream->Read(&sample_response));
    ASSERT_FALSE(sample_stream->Read(&sample_response2));
    REVERB_EXPECT_OK(sample_stream->Finish());
    item.set_times_sampled(2*i + 2);

    sample_response.mutable_entries(0)
        ->mutable_info()
        ->mutable_item()
        ->set_times_sampled(2 * i + 2);
    EXPECT_THAT(sample_response.entries(0),
                testing::EqualsProto(sample_response.entries(1)));
    SampleInfo info = sample_response.entries(0).info();

    info.mutable_item()->clear_inserted_at();
    EXPECT_THAT(info.item(), testing::EqualsProto(item));
    EXPECT_EQ(info.probability(), 1);
    EXPECT_EQ(info.table_size(), 1);
    EXPECT_FALSE(info.rate_limited());

    EXPECT_EQ(sample_response.entries(0).data_size(), 2);
    EXPECT_EQ(sample_response.entries(0).data(0).chunk_key(),
              item.flat_trajectory().columns(0).chunk_slices(0).chunk_key());
    EXPECT_EQ(sample_response.entries(0).data(1).chunk_key(),
              item.flat_trajectory().columns(0).chunk_slices(1).chunk_key());
  }
}

TEST(ReverbServiceImplTest, InsertChunksWithoutItemWorks) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));

  grpc::ClientContext context;
  auto stream = stub.InsertStream(&context);
  ASSERT_TRUE(stream->Write(InsertChunkRequest(1)));
  ASSERT_TRUE(stream->Write(InsertChunkRequest(2)));
  ASSERT_TRUE(stream->WritesDone());
  REVERB_EXPECT_OK(stream->Finish());
}

TEST(ReverbServiceImplTest, InsertSameChunkTwiceWorks) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));

  grpc::ClientContext context;
  auto stream = stub.InsertStream(&context);
  ASSERT_TRUE(stream->Write(InsertChunkRequest(1)));
  ASSERT_TRUE(stream->Write(InsertChunkRequest(1)));
  ASSERT_TRUE(stream->WritesDone());
  REVERB_EXPECT_OK(stream->Finish());
}

TEST(ReverbServiceImplTest, InsertSameChunkInMultiChunkMessageWorks) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));

  grpc::ClientContext context;
  auto stream = stub.InsertStream(&context);
  ASSERT_TRUE(stream->Write(InsertChunkRequest(1)));
  ASSERT_TRUE(stream->Write(InsertMultiChunkRequest({1, 2})));
  ASSERT_TRUE(stream->WritesDone());
  REVERB_EXPECT_OK(stream->Finish());
}

TEST(ReverbServiceImplTest, InsertItemWithoutKeptChunkFails) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));

  grpc::ClientContext context;
  auto stream = stub.InsertStream(&context);
  ASSERT_TRUE(stream->Write(InsertChunkRequest(1)));
  ASSERT_TRUE(stream->Write(InsertChunkRequest(2)));
  ASSERT_TRUE(stream->Write(InsertItemRequest("dist", {1, 2})));
  ASSERT_TRUE(stream->Write(InsertItemRequest("dist", {2, 3})));
  EXPECT_EQ(stream->Finish().error_code(), grpc::StatusCode::INTERNAL);
}

TEST(ReverbServiceImplTest, InsertItemWithKeptChunkWorks) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));

  grpc::ClientContext context;
  auto stream = stub.InsertStream(&context);
  ASSERT_TRUE(stream->Write(InsertChunkRequest(1)));
  ASSERT_TRUE(stream->Write(InsertChunkRequest(2)));
  ASSERT_TRUE(stream->Write(InsertChunkRequest(3)));
  ASSERT_TRUE(stream->Write(InsertItemRequest("dist", {1, 2}, {2, 3})));
  ASSERT_TRUE(stream->Write(InsertItemRequest("dist", {2, 3})));
  ASSERT_TRUE(stream->WritesDone());
  REVERB_EXPECT_OK(stream->Finish());
}

TEST(ReverbServiceImplTest, InsertItemWithNotKeptChunkFails) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));

  grpc::ClientContext context;
  auto stream = stub.InsertStream(&context);
  ASSERT_TRUE(stream->Write(InsertChunkRequest(1)));
  ASSERT_TRUE(stream->Write(InsertChunkRequest(2)));
  ASSERT_TRUE(stream->Write(InsertChunkRequest(3)));
  ASSERT_TRUE(stream->Write(InsertItemRequest("dist", {1, 2}, {2})));
  ASSERT_TRUE(stream->Write(InsertItemRequest("dist", {2, 3})));
  EXPECT_EQ(stream->Finish().error_code(), grpc::StatusCode::INTERNAL);
}

TEST(ReverbServiceImplTest, InsertItemWithMissingChunksFails) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));

  grpc::ClientContext context;
  auto stream = stub.InsertStream(&context);
  ASSERT_TRUE(stream->Write(InsertChunkRequest(1)));
  ASSERT_TRUE(stream->Write(InsertItemRequest("dist", {2})));
  EXPECT_EQ(stream->Finish().error_code(), grpc::StatusCode::INTERNAL);
}

TEST(ReverbServiceImplTest, InsertStreamRespondsWithItemKeys) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));

  grpc::ClientContext context;
  auto stream = stub.InsertStream(&context);
  ASSERT_TRUE(stream->Write(InsertChunkRequest(1)));
  auto first_id = nextId;
  ASSERT_TRUE(stream->Write(
      InsertItemRequest("dist", {1}, {1}, /*send_confirmation=*/true)));
  ASSERT_TRUE(stream->Write(
      InsertItemRequest("dist", {1}, {1}, /*send_confirmation=*/false)));
  ASSERT_TRUE(stream->Write(
      InsertItemRequest("dist", {1}, {}, /*send_confirmation=*/true)));
  ASSERT_TRUE(stream->WritesDone());
  InsertStreamResponse responses[3];
  ASSERT_TRUE(stream->Read(&responses[0]));
  ASSERT_TRUE(stream->Read(&responses[1]));
  ASSERT_FALSE(stream->Read(&responses[2]));
  REVERB_EXPECT_OK(stream->Finish());
  EXPECT_EQ(responses[0].keys(0), first_id);
  EXPECT_EQ(responses[1].keys(0), first_id + 2);
}

TEST(ReverbServiceImplTest, SampleBlocksUntilEnoughInserts) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));
  absl::Notification notification;
  auto thread = internal::StartThread("", [&] {
    grpc::ClientContext context;
    auto stream = stub.SampleStream(&context);
    ASSERT_TRUE(stream->Write(SampleRequest("dist", 1)));
    ASSERT_TRUE(stream->WritesDone());
    SampleStreamResponse response;
    ASSERT_TRUE(stream->Read(&response));
    REVERB_EXPECT_OK(stream->Finish());
    notification.Notify();
  });

  // Blocking because there are no data to sample.
  EXPECT_FALSE(notification.HasBeenNotified());

  // Insert an item.
  grpc::ClientContext context;
  auto stream = stub.InsertStream(&context);
  ASSERT_TRUE(stream->Write(InsertChunkRequest(1)));
  ASSERT_TRUE(stream->Write(InsertItemRequest("dist", {1})));
  ASSERT_TRUE(stream->WritesDone());
  REVERB_EXPECT_OK(stream->Finish());

  // The sample should now complete because there is data to sample.
  notification.WaitForNotification();

  thread = nullptr;  // Joins the thread.
}

TEST(ReverbServiceImplTest, MutateDeletionWorks) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));
  grpc::ClientContext context;
  auto stream = stub.InsertStream(&context);
  ASSERT_TRUE(stream->Write(InsertChunkRequest(1)));
  auto insert_request = InsertItemRequest("dist", {1});
  PrioritizedItem item = insert_request.item().item();
  ASSERT_TRUE(stream->Write(insert_request));
  ASSERT_TRUE(stream->WritesDone());
  REVERB_EXPECT_OK(stream->Finish());

  WaitForTableSize(service->tables()["dist"].get(), 1);

  MutatePrioritiesRequest mutate_request;
  MutatePrioritiesResponse mutate_response;
  mutate_request.set_table("dist");
  mutate_request.add_delete_keys(item.key());
  grpc::ClientContext mutate_context;
  REVERB_EXPECT_OK(
      stub.MutatePriorities(&mutate_context, mutate_request, &mutate_response));

  EXPECT_EQ(service->tables()["dist"]->size(), 0);
}

TEST(ReverbServiceImplTest, AnyCallWithInvalidDistributionFails) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));

  grpc::ClientContext sample_context;
  auto sample_stream = stub.SampleStream(&sample_context);
  ASSERT_TRUE(sample_stream->Write(SampleRequest("invalid", 1)));
  EXPECT_EQ(sample_stream->Finish().error_code(), grpc::StatusCode::NOT_FOUND);

  grpc::ClientContext mutate_context;
  MutatePrioritiesRequest mutate_request;
  MutatePrioritiesResponse mutate_response;
  mutate_request.set_table("invalid");
  EXPECT_EQ(
      stub.MutatePriorities(&mutate_context, mutate_request, &mutate_response)
          .error_code(),
      grpc::StatusCode::NOT_FOUND);

  grpc::ClientContext insert_context;
  auto insert_stream = stub.InsertStream(&insert_context);
  ASSERT_TRUE(insert_stream->Write(InsertChunkRequest(1)));
  ASSERT_TRUE(insert_stream->Write(InsertItemRequest("invalid", {1})));
  EXPECT_EQ(insert_stream->Finish().error_code(), grpc::StatusCode::NOT_FOUND);
}

TEST(ReverbServiceImplTest, ResetWorks) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));

  grpc::ClientContext context;
  auto stream = stub.InsertStream(&context);
  ASSERT_TRUE(stream->Write(InsertChunkRequest(1)));
  ASSERT_TRUE(stream->Write(InsertItemRequest("dist", {1})));
  ASSERT_TRUE(stream->WritesDone());
  REVERB_EXPECT_OK(stream->Finish());

  WaitForTableSize(service->tables()["dist"].get(), 1);

  ResetRequest reset_request;
  reset_request.set_table("dist");
  ResetResponse reset_response;
  grpc::ClientContext reset_context;
  REVERB_ASSERT_OK(stub.Reset(&reset_context, reset_request, &reset_response));

  EXPECT_EQ(service->tables()["dist"]->size(), 0);
}

TEST(ReverbServiceImplTest, ServerInfoWorks) {
  auto service = MakeService(10);
  grpc::CallbackServerContext context;
  grpc::testing::DefaultReactorTestPeer peer(&context);

  ServerInfoRequest server_info_request;
  ServerInfoResponse server_info_response;

  grpc::ServerUnaryReactor* reactor = service->ServerInfo(
      &context, &server_info_request, &server_info_response);

  ASSERT_EQ(reactor, peer.reactor());
  ASSERT_TRUE(peer.test_status_set());
  REVERB_ASSERT_OK(peer.test_status());

  // The probability of these being 0 is 2^{-128}
  EXPECT_NE(std::make_pair(server_info_response.tables_state_id().low(),
                           server_info_response.tables_state_id().high()),
            std::make_pair(uint64_t{0}, uint64_t{0}));

  EXPECT_EQ(server_info_response.table_info_size(), 1);
  const auto& table_info = server_info_response.table_info()[0];

  TableInfo expected_table_info;
  expected_table_info.set_name("dist");
  expected_table_info.mutable_sampler_options()->set_uniform(true);
  expected_table_info.mutable_sampler_options()->set_is_deterministic(false);
  expected_table_info.mutable_remover_options()->set_fifo(true);
  expected_table_info.mutable_remover_options()->set_is_deterministic(true);
  expected_table_info.set_max_size(10);
  expected_table_info.set_current_size(0);
  auto rate_limiter = expected_table_info.mutable_rate_limiter_info();
  rate_limiter->set_samples_per_insert(kSamplesPerInsert);
  rate_limiter->set_min_size_to_sample(kMinSizeToSample);
  rate_limiter->set_min_diff(kMinDiff);
  rate_limiter->set_max_diff(kMaxDiff);
  rate_limiter->mutable_insert_stats()->mutable_completed_wait_time();
  rate_limiter->mutable_insert_stats()->mutable_pending_wait_time();
  rate_limiter->mutable_sample_stats()->mutable_completed_wait_time();
  rate_limiter->mutable_sample_stats()->mutable_pending_wait_time();
  *expected_table_info.mutable_signature() = MakeSignature();

  EXPECT_THAT(table_info, testing::EqualsProto(expected_table_info));
}

TEST(ReverbServiceImplTest, CheckpointCalledWithoutCheckpointer) {
  auto service = MakeService(10);
  grpc::CallbackServerContext context;
  grpc::testing::DefaultReactorTestPeer peer(&context);
  CheckpointRequest request;
  CheckpointResponse response;

  grpc::ServerUnaryReactor* reactor =
      service->Checkpoint(&context, &request, &response);

  ASSERT_EQ(reactor, peer.reactor());
  ASSERT_TRUE(peer.test_status_set());
  EXPECT_EQ(peer.test_status().error_code(),
            grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(ReverbServiceImplTest, CheckpointAndLoadFromCheckpoint) {
  std::string path = getenv("TEST_TMPDIR");
  REVERB_CHECK(tensorflow::Env::Default()->CreateUniqueFileName(&path, "temp"));
  auto service = MakeService(10, CreateDefaultCheckpointer(path));
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));

  // Check that there are no items in the service to begin with.
  EXPECT_EQ(service->tables()["dist"]->size(), 0);

  // Insert an item.
  {
    grpc::ClientContext context;
    auto stream = stub.InsertStream(&context);
    ASSERT_TRUE(stream->Write(InsertChunkRequest(1)));
    ASSERT_TRUE(stream->Write(InsertItemRequest("dist", {1})));
    ASSERT_TRUE(stream->WritesDone());
    REVERB_EXPECT_OK(stream->Finish());
  }

  WaitForTableSize(service->tables()["dist"].get(), 1);

  // Checkpoint the service.
  {
    CheckpointRequest request;
    CheckpointResponse response;
    grpc::ClientContext context;
    REVERB_EXPECT_OK(stub.Checkpoint(&context, request, &response));
  }

  // Create a new service from the checkpoint and check that it has the
  // correct number of items.
  auto loaded_service = MakeService(10, CreateDefaultCheckpointer(path));
  EXPECT_EQ(loaded_service->tables()["dist"]->size(), 1);
}

TEST(ReverbServiceImplTest, InitializeConnectionSuccess) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));

  grpc::ClientContext context;
  auto stream = stub.InitializeConnection(&context);

  InitializeConnectionRequest request;
  request.set_pid(getpid());
  request.set_table_name("dist");
  ASSERT_TRUE(stream->Write(request));

  InitializeConnectionResponse response;
  ASSERT_TRUE(stream->Read(&response));
  ASSERT_NE(response.address(), 0);

  // Verify that we successfully copied the Table shared_ptr.
  auto server_table = service->tables().find("dist")->second;
  auto* client_table_ptr =
      reinterpret_cast<std::shared_ptr<Table>*>(response.address());
  EXPECT_EQ(*client_table_ptr, server_table);

  // Confirm the transfer and close the connection.
  request.set_ownership_transferred(true);
  ASSERT_TRUE(stream->Write(request));
  ASSERT_TRUE(stream->WritesDone());
  REVERB_EXPECT_OK(stream->Finish());
}

TEST(ReverbServiceImplTest, InitializeConnectionTableNotFound) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));

  grpc::ClientContext context;
  auto stream = stub.InitializeConnection(&context);

  InitializeConnectionRequest request;
  request.set_pid(getpid());
  request.set_table_name("not_found_table");
  ASSERT_TRUE(stream->Write(request));

  // There shouldn't be any response to read.
  InitializeConnectionResponse response;
  ASSERT_FALSE(stream->Read(&response));

  // Status should indicate that table wasn't found.
  EXPECT_EQ(stream->Finish().error_code(), grpc::StatusCode::NOT_FOUND);
}

TEST(ReverbServiceImplTest, InitializeConnectionFromOtherProcess) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  std::unique_ptr<grpc::Server> server(
      grpc::ServerBuilder().RegisterService(service.get()).BuildAndStart());
  /* grpc_gen:: */ReverbService::Stub stub(
      server->InProcessChannel(grpc::ChannelArguments()));

  grpc::ClientContext context;
  auto stream = stub.InitializeConnection(&context);

  InitializeConnectionRequest request;
  // Simulate that we are running in a different process.
  request.set_pid(getpid() + 1);
  request.set_table_name("dist");
  ASSERT_TRUE(stream->Write(request));

  // The response should not contain any address.
  InitializeConnectionResponse response;
  ASSERT_TRUE(stream->Read(&response));
  EXPECT_EQ(response.address(), 0);

  // No more actions are expected and stream should be closable without errors.
  REVERB_EXPECT_OK(stream->Finish());
}

// TODO(b/179142085): Add more tests for the InsertWorker
TEST(InsertWorkerTest, InsertWorkerReturnsCorrectStats) {
  auto insert_worker = absl::make_unique<InsertWorker>(
      /*num_threads=*/1, /*max_queue_size_to_warn=*/3, "TestWorker");
  Table::Item item;
  item.item.set_table("my_table");
  absl::BlockingCounter counter(2);
  for (int i = 0; i < 2; i++) {
    InsertTaskInfo task_info;
    task_info.item = item;
    insert_worker->Schedule(
        task_info,
        [&counter](InsertTaskInfo task_info, const absl::Status& status,
                                bool enough_queue_slots) {
          counter.DecrementCount();
        });
  }
  counter.Wait();

  while (insert_worker->GetThreadStats()[0].num_tasks_processed < 2) {}

  auto stats = insert_worker->GetThreadStats();
  ASSERT_THAT(stats, ::testing::SizeIs(1));
  EXPECT_EQ(stats[0].current_task_id, 1);
  EXPECT_EQ(stats[0].num_tasks_processed, 2);
  EXPECT_THAT(stats[0].current_task_info, ::testing::HasSubstr("my_table"));
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
