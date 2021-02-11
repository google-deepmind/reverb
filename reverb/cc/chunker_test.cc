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

#include "reverb/cc/chunker.h"

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "reverb/cc/testing/tensor_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"

namespace deepmind {
namespace reverb {
namespace {

using ::testing::ElementsAre;

const auto kIntSpec = internal::TensorSpec{"0", tensorflow::DT_INT32, {1}};
const auto kFloatSpec = internal::TensorSpec{"0", tensorflow::DT_FLOAT, {1}};

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

TEST(CellRef, IsReady) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, 2, 5);

  std::weak_ptr<CellRef> ref;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec), {1, 0}, &ref));

  // Chunk is not finalized yet.
  EXPECT_FALSE(ref.lock()->IsReady());

  // Force chunk creation.
  REVERB_ASSERT_OK(chunker->Flush());
  EXPECT_TRUE(ref.lock()->IsReady());
}

TEST(CellRef, GetDataFromChunkerBuffer) {
  internal::TensorSpec spec = {"0", tensorflow::DT_INT32, {3, 3}};
  auto chunker = std::make_shared<Chunker>(spec,
                                           /*max_chunk_length=*/2,
                                           /*num_keep_alive_refs=*/2);

  std::weak_ptr<CellRef> ref;
  auto want = MakeConstantTensor<tensorflow::DT_INT32>({3, 3}, 5);
  REVERB_ASSERT_OK(chunker->Append(want, {1, 0}, &ref));

  // Chunk is not finalized yet so `GetData` must read from Chunker buffer.
  EXPECT_FALSE(ref.lock()->IsReady());

  tensorflow::Tensor got;
  REVERB_ASSERT_OK(ref.lock()->GetData(&got));
  test::ExpectTensorEqual<tensorflow::int32>(got, want);
}

TEST(CellRef, GetDataFromChunk) {
  internal::TensorSpec spec = {"0", tensorflow::DT_FLOAT, {3, 3}};
  auto chunker = std::make_shared<Chunker>(spec,
                                           /*max_chunk_length=*/2,
                                           /*num_keep_alive_refs=*/2);

  // Take two steps to finalize the chunk.
  std::weak_ptr<CellRef> first;
  auto first_want = MakeConstantTensor<tensorflow::DT_FLOAT>({3, 3}, 1);
  REVERB_ASSERT_OK(chunker->Append(first_want, {1, 0}, &first));

  std::weak_ptr<CellRef> second;
  auto second_want = MakeConstantTensor<tensorflow::DT_FLOAT>({3, 3}, 2);
  REVERB_ASSERT_OK(chunker->Append(second_want, {1, 1}, &second));

  // Both steps should be finalized.
  EXPECT_TRUE(first.lock()->IsReady());
  EXPECT_TRUE(second.lock()->IsReady());

  // Check that the data is correct when reading it back from the chunk.
  tensorflow::Tensor first_got;
  REVERB_ASSERT_OK(first.lock()->GetData(&first_got));
  test::ExpectTensorEqual<float>(first_got, first_want);

  tensorflow::Tensor second_got;
  REVERB_ASSERT_OK(second.lock()->GetData(&second_got));
  test::ExpectTensorEqual<float>(second_got, second_want);
}

TEST(Chunker, AppendValidatesSpecDtype) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/2,
                                           /*num_keep_alive_refs=*/5);

  std::weak_ptr<CellRef> ref;
  auto status = chunker->Append(MakeTensor(kFloatSpec), {1, 0}, &ref);

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  absl::StrCat("Tensor of wrong dtype provided for column 0. "
                               "Got float but expected ",
                               Int32Str(), ".")));
}

TEST(Chunker, AppendValidatesSpecShape) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/2,
                                           /*num_keep_alive_refs=*/5);

  std::weak_ptr<CellRef> ref;
  auto status = chunker->Append(
      MakeTensor(internal::TensorSpec{kIntSpec.name, kIntSpec.dtype, {2}}),
      {/*episode_id=*/1, /*step=*/0}, &ref);

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Tensor of incompatible shape provided for column 0. "
                  "Got [2] which is incompatible with [1]."));
}

TEST(Chunker, AppendFlushesOnMaxChunkLength) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/2,
                                           /*num_keep_alive_refs=*/5);

  // Buffer is not full after first step.
  std::weak_ptr<CellRef> first;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/0}, &first));
  EXPECT_FALSE(first.lock()->IsReady());

  // Second step should trigger flushing of buffer.
  std::weak_ptr<CellRef> second;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/1}, &second));
  EXPECT_TRUE(first.lock()->IsReady());
  EXPECT_TRUE(second.lock()->IsReady());
}

TEST(Chunker, Flush) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/2,
                                           /*num_keep_alive_refs=*/5);
  std::weak_ptr<CellRef> ref;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/0}, &ref));
  EXPECT_FALSE(ref.lock()->IsReady());
  REVERB_ASSERT_OK(chunker->Flush());
  EXPECT_TRUE(ref.lock()->IsReady());
}

TEST(Chunker, ChunkHasBatchDim) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/2,
                                           /*num_keep_alive_refs=*/5);

  // Add two data items to trigger the finalization.
  std::weak_ptr<CellRef> ref;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/0}, &ref));
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/1}, &ref));
  ASSERT_TRUE(ref.lock()->IsReady());
  EXPECT_THAT(ref.lock()->GetChunk()->data().tensors(0).tensor_shape(),
              testing::EqualsProto("dim { size: 2} dim { size: 1}"));

  // The batch dim is added even if it only contains a single step.
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/0}, &ref));
  REVERB_ASSERT_OK(chunker->Flush());
  ASSERT_TRUE(ref.lock()->IsReady());
  EXPECT_THAT(ref.lock()->GetChunk()->data().tensors(0).tensor_shape(),
              testing::EqualsProto("dim { size: 1} dim { size: 1}"));
}

TEST(Chunker, DeletesRefsWhenMageAgeExceeded) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/2,
                                           /*num_keep_alive_refs=*/3);

  std::weak_ptr<CellRef> first;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/0}, &first));
  EXPECT_FALSE(first.expired());

  std::weak_ptr<CellRef> second;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/1}, &second));
  EXPECT_FALSE(first.expired());
  EXPECT_FALSE(second.expired());

  std::weak_ptr<CellRef> third;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/2}, &third));
  EXPECT_FALSE(first.expired());
  EXPECT_FALSE(second.expired());
  EXPECT_FALSE(third.expired());

  std::weak_ptr<CellRef> fourth;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/3}, &fourth));
  EXPECT_TRUE(first.expired());
  EXPECT_FALSE(second.expired());
  EXPECT_FALSE(third.expired());
  EXPECT_FALSE(fourth.expired());
}

TEST(Chunker, GetKeepKeys) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/2,
                                           /*num_keep_alive_refs=*/2);

  std::weak_ptr<CellRef> first;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/0}, &first));
  EXPECT_THAT(chunker->GetKeepKeys(), ElementsAre(first.lock()->chunk_key()));

  // The second ref will belong to the same chunk.
  std::weak_ptr<CellRef> second;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/1}, &second));
  EXPECT_THAT(chunker->GetKeepKeys(), ElementsAre(first.lock()->chunk_key()));

  // The third ref will belong to a new chunk. The first ref is now expired but
  // since the second ref belong to the same chunk we expect the chunker to tell
  // us to keep both chunks around.
  std::weak_ptr<CellRef> third;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/2}, &third));
  EXPECT_THAT(chunker->GetKeepKeys(), ElementsAre(second.lock()->chunk_key(),
                                                  third.lock()->chunk_key()));

  // Adding a fourth value results in the second one expiring. The only chunk
  // which should be kept thus is the one referenced by the third and fourth.
  std::weak_ptr<CellRef> fourth;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/3}, &fourth));
  EXPECT_THAT(chunker->GetKeepKeys(), ElementsAre(third.lock()->chunk_key()));
}

TEST(Chunker, ResetClearsRefs) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/2,
                                           /*num_keep_alive_refs=*/2);

  std::weak_ptr<CellRef> first;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/0}, &first));
  std::weak_ptr<CellRef> second;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/1}, &second));

  // Before resetting both references are alive.
  EXPECT_FALSE(first.expired());
  EXPECT_FALSE(second.expired());

  // After resetting both references are dead.
  chunker->Reset();
  EXPECT_TRUE(first.expired());
  EXPECT_TRUE(second.expired());
}

TEST(Chunker, ResetRefreshesChunkKey) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/2,
                                           /*num_keep_alive_refs=*/2);

  std::weak_ptr<CellRef> first;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/0}, &first));

  // Extract key since the `CellRef` will expire when we reset the
  // `Chunker`.
  uint64_t first_chunk_key = first.lock()->chunk_key();

  chunker->Reset();

  // Take a second step now that the Chunker have been reseted. Note that since
  // `max_chunk_length` hasn't been reached we would expect the second step to
  // be part of the same chunk if `Reset` wasn't called in between.
  std::weak_ptr<CellRef> second;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/1}, &second));

  EXPECT_NE(second.lock()->chunk_key(), first_chunk_key);
}

TEST(Chunker, ResetRefreshesOffset) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/2,
                                           /*num_keep_alive_refs=*/2);

  std::weak_ptr<CellRef> first;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/0}, &first));

  chunker->Reset();

  // Take a second step now that the Chunker have been reseted. Note that since
  // `max_chunk_length` hasn't been reached we would expect the second step to
  // be part of the same chunk if `Reset` wasn't called in between.
  std::weak_ptr<CellRef> second;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/1}, &second));

  EXPECT_EQ(second.lock()->offset(), 0);
}

TEST(Chunker, AppendRequiresSameEpisode) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/3,
                                           /*num_keep_alive_refs=*/3);

  // Add two steps referencing two different episodes.
  std::weak_ptr<CellRef> first;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/0}, &first));
  std::weak_ptr<CellRef> second;
  auto status = chunker->Append(MakeTensor(kIntSpec),
                                {/*episode_id=*/2, /*step=*/0}, &second);

  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr(
          "Chunker::Append called with new episode when buffer non empty."));
}

TEST(Chunker, AppendRequiresEpisodeStepIncreases) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/3,
                                           /*num_keep_alive_refs=*/3);

  // Add two steps referencing two different episodes.
  std::weak_ptr<CellRef> first;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/5}, &first));

  // Same step index.
  std::weak_ptr<CellRef> eq;
  auto eq_status = chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/5}, &eq);

  EXPECT_EQ(eq_status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(
      std::string(eq_status.message()),
      ::testing::HasSubstr("Chunker::Append called with an episode step "
                           "which was not greater than already observed."));

  // Smaller step index.
  std::weak_ptr<CellRef> lt;
  auto lt_status = chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/3}, &lt);

  EXPECT_EQ(lt_status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(
      std::string(lt_status.message()),
      ::testing::HasSubstr("Chunker::Append called with an episode step "
                           "which was not greater than already observed."));
}

TEST(Chunker, NonSparseEpisodeRange) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/5,
                                           /*num_keep_alive_refs=*/5);

  // Append five consecutive steps.
  std::weak_ptr<CellRef> step;
  for (int i = 0; i < 5; i++) {
    REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                     {/*episode_id=*/1, /*step=*/i}, &step));
  }

  // Check that the range is non sparse.
  ASSERT_FALSE(step.expired());
  ASSERT_TRUE(step.lock()->IsReady());
  EXPECT_THAT(step.lock()->GetChunk()->sequence_range(),
              testing::EqualsProto("episode_id: 1 start: 0 end: 4"));
}

TEST(Chunker, SparseEpisodeRange) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/5,
                                           /*num_keep_alive_refs=*/5);

  // Append five steps with a stride of 2.
  std::weak_ptr<CellRef> step;
  for (int i = 0; i < 5; i++) {
    REVERB_ASSERT_OK(chunker->Append(
        MakeTensor(kIntSpec), {/*episode_id=*/33, /*step=*/i * 2}, &step));
  }

  // Check that the range is non sparse.
  ASSERT_FALSE(step.expired());
  ASSERT_TRUE(step.lock()->IsReady());
  EXPECT_THAT(
      step.lock()->GetChunk()->sequence_range(),
      testing::EqualsProto("episode_id: 33 start: 0 end: 8 sparse: true"));
}

TEST(Chunker, ApplyConfigChangesMaxChunkLength) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/5,
                                           /*num_keep_alive_refs=*/5);

  // Reconfigure the chunk_length to be 1 instead of 5.
  REVERB_ASSERT_OK(
      chunker->ApplyConfig(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/5));

  // Appending should now result in chunks being created with each step.
  std::weak_ptr<CellRef> step;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/0}, &step));
  ASSERT_FALSE(step.expired());
  ASSERT_TRUE(step.lock()->IsReady());
  EXPECT_THAT(step.lock()->GetChunk()->sequence_range(),
              testing::EqualsProto("episode_id: 1 start: 0 end: 0"));
}

TEST(Chunker, ApplyConfigChangesNumKeepAliveRefs) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/1,
                                           /*num_keep_alive_refs=*/1);

  // Reconfigure num_keep_alive_refs to be 2 instead of 1.
  REVERB_ASSERT_OK(
      chunker->ApplyConfig(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/2));

  // The last two steps should now be alive instead of only the last one.
  std::weak_ptr<CellRef> first;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/0}, &first));
  ASSERT_FALSE(first.expired());

  std::weak_ptr<CellRef> second;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/1}, &second));
  ASSERT_FALSE(first.expired());
  ASSERT_FALSE(second.expired());

  std::weak_ptr<CellRef> third;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/2}, &third));
  ASSERT_TRUE(first.expired());
  ASSERT_FALSE(second.expired());
  ASSERT_FALSE(third.expired());
}

TEST(Chunker, ApplyConfigRequireBufferToBeEmpty) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/5,
                                           /*num_keep_alive_refs=*/5);

  // Append a step which is not finalized since max_chunk_length is 2.
  std::weak_ptr<CellRef> step;
  REVERB_ASSERT_OK(chunker->Append(MakeTensor(kIntSpec),
                                   {/*episode_id=*/1, /*step=*/0}, &step));

  auto status =
      chunker->ApplyConfig(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/5);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("Flush must be called before ApplyConfig."));

  // Flushing and retrying the same configure call should succeed.
  REVERB_ASSERT_OK(chunker->Flush());
  REVERB_EXPECT_OK(
      chunker->ApplyConfig(/*max_chunk_length=*/1, /*num_keep_alive_refs=*/5));
}

TEST(Chunker, ApplyConfigRejectsInvalidOptions) {
  auto chunker = std::make_shared<Chunker>(kIntSpec, /*max_chunk_length=*/5,
                                           /*num_keep_alive_refs=*/5);
  std::vector<std::pair<int, int>> invalid_options = {
      {0, 5},   // max_chunk_length must be > 0.
      {-1, 5},  // max_chunk_length must be > 0.
      {5, 0},   // num_keep_alive_refs must be > 0.
      {5, -1},  // num_keep_alive_refs must be > 0.
      {6, 5},   // num_keep_alive_refs must be >= max_chunk_length.
  };
  for (const auto [max_chunk_length, num_keep_alive_refs] : invalid_options) {
    EXPECT_EQ(
        chunker->ApplyConfig(max_chunk_length, num_keep_alive_refs).code(),
        absl::StatusCode::kInvalidArgument);
  }
}

TEST(ValidateChunkerOptions, Valid) {
  REVERB_EXPECT_OK(ValidateChunkerOptions(/*max_chunk_length=*/2,
                                          /*num_keep_alive_refs=*/2));
}

TEST(ValidateChunkerOptions, ZeroMaxChunkLength) {
  auto status =
      ValidateChunkerOptions(/*max_chunk_length=*/0, /*num_keep_alive_refs=*/2);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("max_chunk_length must be > 0 but got 0."));
}

TEST(ValidateChunkerOptions, NegativeMaxChunkLength) {
  auto status = ValidateChunkerOptions(/*max_chunk_length=*/-1,
                                       /*num_keep_alive_refs=*/2);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("max_chunk_length must be > 0 but got -1."));
}

TEST(ValidateChunkerOptions, ZeroNumKeepAliveRefs) {
  auto status =
      ValidateChunkerOptions(/*max_chunk_length=*/2, /*num_keep_alive_refs=*/0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr("num_keep_alive_refs must be > 0 but got 0."));
}

TEST(ValidateChunkerOptions, NegativeNumKeepAliveRefs) {
  auto status = ValidateChunkerOptions(/*max_chunk_length=*/2,
                                       /*num_keep_alive_refs=*/-1);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr("num_keep_alive_refs must be > 0 but got -1."));
}

TEST(ValidateChunkerOptions, NumKeepAliveLtMaxChunkLength) {
  auto status =
      ValidateChunkerOptions(/*max_chunk_length=*/6, /*num_keep_alive_refs=*/5);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "num_keep_alive_refs (5) must be >= max_chunk_length (6)."));
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
