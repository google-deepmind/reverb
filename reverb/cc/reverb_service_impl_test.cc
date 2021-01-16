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
#include <list>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/notification.h"
#include "absl/types/optional.h"
#include "reverb/cc/platform/checkpointing.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/selectors/uniform.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace deepmind {
namespace reverb {
namespace {

const int64_t kMinSizeToSample = 1;
const double kSamplesPerInsert = 1.0;
const double kMinDiff = -DBL_MAX;
const double kMaxDiff = DBL_MAX;

int64_t nextId = 1;

class FakeInsertStream
    : public grpc::ServerReaderWriterInterface<InsertStreamResponse,
                                               InsertStreamRequest> {
 public:
  void AddChunk(int64_t key) {
    InsertStreamRequest request;
    request.mutable_chunk()->set_chunk_key(key);
    read_buffer_.push_back(std::move(request));
  }

  PrioritizedItem AddItem(absl::string_view table,
                          const std::vector<int64_t>& sequence_chunks,
                          const std::vector<int64_t>& keep_chunks = {},
                          bool send_confirmation = false) {
    PrioritizedItem item;
    item.set_key(nextId++);
    item.set_table(table.data(), table.size());
    *item.mutable_chunk_keys() = {sequence_chunks.begin(),
                                  sequence_chunks.end()};
    if (!sequence_chunks.empty()) {
      item.mutable_sequence_range()->set_offset(0);
      item.mutable_sequence_range()->set_length(100);
    }

    InsertStreamRequest request;
    *request.mutable_item()->mutable_keep_chunk_keys() = {keep_chunks.begin(),
                                                          keep_chunks.end()};
    *request.mutable_item()->mutable_item() = item;
    request.mutable_item()->set_send_confirmation(send_confirmation);
    read_buffer_.push_back(std::move(request));
    return item;
  }

  bool Read(InsertStreamRequest* request) override {
    if (read_buffer_.empty()) return false;
    *request = read_buffer_.front();
    read_buffer_.pop_front();
    return true;
  }

  bool Write(const InsertStreamResponse& response,
             grpc::WriteOptions options) override {
    responses_.push_back(response);
    return true;
  }

  void SendInitialMetadata() override {}
  bool NextMessageSize(uint32_t*) override { return false; }

  std::vector<InsertStreamResponse> responses() const { return responses_; }

 private:
  std::list<InsertStreamRequest> read_buffer_;
  std::vector<InsertStreamResponse> responses_;
};

class FakeSampleStream
    : public grpc::ServerReaderWriterInterface<SampleStreamResponse,
                                               SampleStreamRequest> {
 public:
  explicit FakeSampleStream() = default;

  const std::vector<SampleStreamResponse>& responses() { return buffer_; }
  const grpc::WriteOptions last_options() { return options_; }

  bool Write(const SampleStreamResponse& response,
             grpc::WriteOptions options) override {
    buffer_.push_back(response);
    options_ = options;
    return true;
  }

  bool Read(SampleStreamRequest* request) override {
    if (requests_.empty()) return false;
    request->set_table(requests_.front().table());
    request->set_num_samples(requests_.front().num_samples());
    request->set_flexible_batch_size(-1);
    requests_.pop_front();
    return true;
  }

  bool NextMessageSize(uint32_t* sz) override {
    if (!requests_.empty()) *sz = requests_.front().ByteSizeLong();
    return !requests_.empty();
  }

  void AddRequest(std::string table, int num_samples) {
    SampleStreamRequest request;
    request.set_table(std::move(table));
    request.set_num_samples(num_samples);
    requests_.push_back(std::move(request));
  }

  void SendInitialMetadata() override {}

 private:
  std::list<SampleStreamRequest> requests_;
  std::vector<SampleStreamResponse> buffer_;
  grpc::WriteOptions options_;
};

tensorflow::StructuredValue MakeSignature() {
  tensorflow::StructuredValue signature;
  auto* tensor_spec = signature.mutable_tensor_spec_value();
  tensor_spec->set_name("item0");
  tensorflow::TensorShape().AsProto(tensor_spec->mutable_shape());
  tensor_spec->set_dtype(tensorflow::DT_INT32);
  return signature;
}

std::unique_ptr<ReverbServiceImpl> MakeService(
    int max_size, std::unique_ptr<Checkpointer> checkpointer) {
  std::vector<std::shared_ptr<Table>> tables;

  tables.push_back(absl::make_unique<Table>(
      "dist", absl::make_unique<UniformSelector>(),
      absl::make_unique<FifoSelector>(), max_size, 0,
      absl::make_unique<RateLimiter>(kSamplesPerInsert, kMinSizeToSample,
                                     kMinDiff, kMaxDiff),
      /*extensions=*/
      std::vector<std::shared_ptr<TableExtension>>{},
      /*signature=*/absl::make_optional(MakeSignature())));
  std::unique_ptr<ReverbServiceImpl> service;
  TF_CHECK_OK(ReverbServiceImpl::Create(std::move(tables),
                                        std::move(checkpointer), &service));
  return service;
}

std::unique_ptr<ReverbServiceImpl> MakeService(int max_size) {
  return MakeService(max_size, nullptr);
}

TEST(ReverbServiceImplTest, SampleAfterInsertWorks) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);

  FakeInsertStream stream;
  stream.AddChunk(1);
  stream.AddChunk(2);
  stream.AddChunk(3);
  PrioritizedItem item = stream.AddItem("dist", {2, 3});
  ASSERT_TRUE(service->InsertStreamInternal(nullptr, &stream).ok());

  for (int i = 0; i < 5; i++) {
    FakeSampleStream stream;
    stream.AddRequest("dist", 1);

    grpc::ServerContext context;
    ASSERT_TRUE(service->SampleStreamInternal(&context, &stream).ok());
    ASSERT_EQ(stream.responses().size(), 2);

    item.set_times_sampled(i + 1);

    SampleInfo info = stream.responses()[0].info();
    info.mutable_item()->clear_inserted_at();
    EXPECT_THAT(info.item(), testing::EqualsProto(item));
    EXPECT_EQ(info.probability(), 1);
    EXPECT_EQ(info.table_size(), 1);

    EXPECT_EQ(stream.responses()[0].data().chunk_key(), item.chunk_keys(0));
    EXPECT_EQ(stream.responses()[1].data().chunk_key(), item.chunk_keys(1));
    EXPECT_TRUE(stream.last_options().get_no_compression());
  }
}

TEST(ReverbServiceImplTest, InsertChunksWithoutItemWorks) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  grpc::ServerContext context;

  FakeInsertStream stream;
  stream.AddChunk(1);
  stream.AddChunk(2);
  EXPECT_OK(service->InsertStreamInternal(&context, &stream));
}

TEST(ReverbServiceImplTest, InsertSameChunkTwiceWorks) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  grpc::ServerContext context;

  FakeInsertStream stream;
  stream.AddChunk(1);
  stream.AddChunk(1);
  EXPECT_OK(service->InsertStreamInternal(&context, &stream));
}

TEST(ReverbServiceImplTest, InsertItemWithoutKeptChunkFails) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  grpc::ServerContext context;

  FakeInsertStream stream;
  stream.AddChunk(1);
  stream.AddChunk(2);
  stream.AddItem("dist", {1, 2});
  stream.AddItem("dist", {2, 3});
  EXPECT_EQ(service->InsertStreamInternal(&context, &stream).error_code(),
            grpc::StatusCode::INTERNAL);
}

TEST(ReverbServiceImplTest, InsertItemWithKeptChunkWorks) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  grpc::ServerContext context;

  FakeInsertStream stream;
  stream.AddChunk(1);
  stream.AddChunk(2);
  stream.AddItem("dist", {1, 2}, {2});
  stream.AddItem("dist", {2, 3});
  EXPECT_EQ(service->InsertStreamInternal(&context, &stream).error_code(),
            grpc::StatusCode::INTERNAL);
}

TEST(ReverbServiceImplTest, InsertItemWithMissingChunksFails) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  grpc::ServerContext context;

  FakeInsertStream stream;
  stream.AddChunk(1);
  stream.AddItem("dist", {2});
  EXPECT_EQ(service->InsertStreamInternal(&context, &stream).error_code(),
            grpc::StatusCode::INTERNAL);
}

TEST(ReverbServiceImplTest, InsertStreamRespondsWithItemKeys) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  grpc::ServerContext context;

  FakeInsertStream stream;
  stream.AddChunk(1);
  auto first_id = nextId;
  stream.AddItem("dist", {1}, {1}, /*send_confirmation=*/true);
  stream.AddItem("dist", {1}, {1}, /*send_confirmation=*/false);
  stream.AddItem("dist", {1}, {}, /*send_confirmation=*/true);
  EXPECT_OK(service->InsertStreamInternal(&context, &stream));
  EXPECT_THAT(stream.responses(), ::testing::SizeIs(2));
  EXPECT_EQ(stream.responses()[0].key(), first_id);
  EXPECT_EQ(stream.responses()[1].key(), first_id + 2);
}

TEST(ReverbServiceImplTest, SampleBlocksUntilEnoughInserts) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  absl::Notification notification;
  auto thread = internal::StartThread("", [&] {
    FakeSampleStream stream;
    stream.AddRequest("dist", 1);
    grpc::ServerContext context;
    EXPECT_OK(service->SampleStreamInternal(&context, &stream));
    notification.Notify();
  });

  // Blocking because there are no data to sample.
  EXPECT_FALSE(notification.HasBeenNotified());

  // Insert an item.
  FakeInsertStream stream;
  stream.AddChunk(1);
  stream.AddItem("dist", {1});
  ASSERT_TRUE(service->InsertStreamInternal(nullptr, &stream).ok());

  // The sample should now complete because there is data to sample.
  notification.WaitForNotification();

  thread = nullptr;  // Joins the thread.
}

TEST(ReverbServiceImplTest, MutateDeletionWorks) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);

  FakeInsertStream stream;
  stream.AddChunk(1);
  PrioritizedItem item = stream.AddItem("dist", {1});
  ASSERT_TRUE(service->InsertStreamInternal(nullptr, &stream).ok());

  EXPECT_EQ(service->tables()["dist"]->size(), 1);

  MutatePrioritiesRequest mutate_request;
  mutate_request.set_table("dist");
  mutate_request.add_delete_keys(item.key());
  EXPECT_OK(service->MutatePriorities(nullptr, &mutate_request, nullptr));

  EXPECT_EQ(service->tables()["dist"]->size(), 0);
}

TEST(ReverbServiceImplTest, AnyCallWithInvalidDistributionFails) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);
  grpc::ServerContext context;

  FakeSampleStream sample_stream;
  sample_stream.AddRequest("invalid", 1);
  EXPECT_EQ(
      service->SampleStreamInternal(&context, &sample_stream).error_code(),
      grpc::StatusCode::NOT_FOUND);

  MutatePrioritiesRequest mutate_request;
  mutate_request.set_table("invalid");
  EXPECT_EQ(
      service->MutatePriorities(nullptr, &mutate_request, nullptr).error_code(),
      grpc::StatusCode::NOT_FOUND);

  FakeInsertStream stream;
  stream.AddChunk(1);
  stream.AddItem("invalid", {1});
  EXPECT_EQ(service->InsertStreamInternal(nullptr, &stream).error_code(),
            grpc::StatusCode::NOT_FOUND);
}

TEST(ReverbServiceImplTest, ResetWorks) {
  std::unique_ptr<ReverbServiceImpl> service = MakeService(10);

  FakeInsertStream stream;
  stream.AddChunk(1);
  PrioritizedItem item = stream.AddItem("dist", {1});
  ASSERT_TRUE(service->InsertStreamInternal(nullptr, &stream).ok());

  EXPECT_EQ(service->tables()["dist"]->size(), 1);

  ResetRequest reset_request;
  reset_request.set_table("dist");
  ResetResponse reset_response;
  ASSERT_TRUE(service->Reset(nullptr, &reset_request, &reset_response).ok());

  EXPECT_EQ(service->tables()["dist"]->size(), 0);
}

TEST(ReverbServiceImplTest, ServerInfoWorks) {
  auto service = MakeService(10);

  ServerInfoRequest server_info_request;
  ServerInfoResponse server_info_response;
  ASSERT_TRUE(
      service->ServerInfo(nullptr, &server_info_request, &server_info_response)
          .ok());

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
  CheckpointRequest request;
  CheckpointResponse response;

  EXPECT_EQ(service->Checkpoint(nullptr, &request, &response).error_code(),
            grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(ReverbServiceImplTest, CheckpointAndLoadFromCheckpoint) {
  std::string path = getenv("TEST_TMPDIR");
  REVERB_CHECK(tensorflow::Env::Default()->CreateUniqueFileName(&path, "temp"));
  auto service = MakeService(10, CreateDefaultCheckpointer(path));

  // Check that there are no items in the service to begin with.
  EXPECT_EQ(service->tables()["dist"]->size(), 0);

  // Insert an item.
  {
    FakeInsertStream stream;
    stream.AddChunk(1);
    stream.AddItem("dist", {1});
    ASSERT_TRUE(service->InsertStreamInternal(nullptr, &stream).ok());
  }

  EXPECT_EQ(service->tables()["dist"]->size(), 1);

  // Checkpoint the service.
  {
    CheckpointRequest request;
    CheckpointResponse response;
    grpc::ServerContext context;
    EXPECT_OK(service->Checkpoint(nullptr, &request, &response));
  }

  // Create a new service from the checkpoint and check that it has the correct
  // number of items.
  auto loaded_service = MakeService(10, CreateDefaultCheckpointer(path));
  EXPECT_EQ(loaded_service->tables()["dist"]->size(), 1);
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
