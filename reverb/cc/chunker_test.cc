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

#include <deque>
#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/schema.pb.h"
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

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Return;

const auto kIntSpec = internal::TensorSpec{"0", tensorflow::DT_INT32, {1}};
const auto kFloatSpec = internal::TensorSpec{"0", tensorflow::DT_FLOAT, {1}};
const auto kLargeFloatSpec =
    internal::TensorSpec{"0", tensorflow::DT_FLOAT, {100, 100}};

inline std::string Int32Str() {
  return tensorflow::DataTypeString(tensorflow::DT_INT32);
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

template <tensorflow::DataType dtype>
tensorflow::Tensor MakeZeroTensor(const internal::TensorSpec& spec) {
  REVERB_CHECK(tensorflow::kRealNumberTypes.Contains(spec.dtype));
  tensorflow::TensorShape shape;
  REVERB_CHECK(spec.shape.AsTensorShape(&shape));
  return MakeConstantTensor<dtype>(shape, 0);
}

template <tensorflow::DataType dtype>
tensorflow::Tensor MakeRandomTensor(
    const tensorflow::TensorShape& shape,
    typename tensorflow::EnumToDataType<dtype>::Type low,
    typename tensorflow::EnumToDataType<dtype>::Type high) {
  tensorflow::Tensor tensor(dtype, shape);
  for (int i = 0; i < tensor.NumElements(); i++) {
    tensor.flat<typename tensorflow::EnumToDataType<dtype>::Type>().data()[i] =
        absl::Uniform<typename tensorflow::EnumToDataType<dtype>::Type>(
            absl::BitGen(), low, high);
  }
  return tensor;
}

tensorflow::Tensor AddBatchDimension(const tensorflow::Tensor& tensor){
  tensorflow::TensorShape shape = tensor.shape();
  shape.InsertDim(0, 1);
  tensorflow::Tensor batched_tensor(tensor.dtype(), shape);
  REVERB_CHECK(batched_tensor.CopyFrom(tensor, shape));
  return batched_tensor;
}

std::shared_ptr<Chunker> MakeChunker(internal::TensorSpec spec,
                                     int max_chunk_length,
                                     int num_keep_alive_refs,
                                     bool delta_encode = false,
                                     bool disable_compression = false) {
  if (disable_compression){
    return std::make_shared<Chunker>(
      std::move(spec),
       std::make_shared<NeverCompressChunkerOptions>(num_keep_alive_refs));
  }
  return std::make_shared<Chunker>(
      std::move(spec),
      std::make_shared<ConstantChunkerOptions>(
          max_chunk_length, num_keep_alive_refs, delta_encode));
}

class MockChunkerOptions : public ChunkerOptions {
 public:
  MOCK_METHOD(int, GetMaxChunkLength, (), (const, override));
  MOCK_METHOD(int, GetNumKeepAliveRefs, (), (const, override));
  MOCK_METHOD(bool, GetDeltaEncode, (), (const, override));
  MOCK_METHOD(bool, GetCompressionDisabled, (), (const, override));
  MOCK_METHOD(absl::Status, OnItemFinalized,
              (const PrioritizedItem& item,
               absl::Span<const std::shared_ptr<CellRef>> refs),
              (override));
  MOCK_METHOD(std::shared_ptr<ChunkerOptions>, Clone, (), (const, override));
};

TEST(CellRef, IsReady) {
  auto chunker = MakeChunker(kIntSpec, 2, 5);

  std::weak_ptr<CellRef> ref;
  REVERB_ASSERT_OK(chunker->Append(
      MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec), {1, 0}, &ref));

  // Chunk is not finalized yet.
  EXPECT_FALSE(ref.lock()->IsReady());

  // Force chunk creation.
  REVERB_ASSERT_OK(chunker->Flush());
  EXPECT_TRUE(ref.lock()->IsReady());
}

TEST(CellRef, GetDataFromChunkerBuffer) {
  internal::TensorSpec spec = {"0", tensorflow::DT_INT32, {3, 3}};
  auto chunker = MakeChunker(spec,
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


TEST(CellRef, GetDataFromUncompressdChunkerBuffer) {
  internal::TensorSpec spec = {"0", tensorflow::DT_INT32, {3, 3}};
  auto chunker = MakeChunker(spec,
                             /*max_chunk_length=*/2,
                             /*num_keep_alive_refs=*/2,
                            /*delta_encode=*/false,
                            /*disable_compression=*/true);

  std::weak_ptr<CellRef> ref;
  auto want = MakeConstantTensor<tensorflow::DT_INT32>({3, 3}, 5);
  REVERB_ASSERT_OK(chunker->Append(want, {1, 0}, &ref));

  // Chunk is never created so `GetData` must read from the buffer.
  EXPECT_FALSE(ref.lock()->IsReady());

  tensorflow::Tensor got;
  REVERB_ASSERT_OK(ref.lock()->GetData(&got));

  test::ExpectTensorEqual<tensorflow::int32>(got, AddBatchDimension(want));
}

TEST(CellRef, GetDataFromChunk) {
  for (bool delta_encode : {true, false}) {
    internal::TensorSpec spec = {"0", tensorflow::DT_FLOAT, {3, 3}};
    auto chunker =
        MakeChunker(spec,
                    /*max_chunk_length=*/2,
                    /*num_keep_alive_refs=*/2, /*delta_encode=*/delta_encode);

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
}


TEST(CellRef, GetDataFromUncompressedBufferNeverCreatesChunks) {
    internal::TensorSpec spec = {"0", tensorflow::DT_FLOAT, {3, 3}};
    auto chunker =
        MakeChunker(spec,
                    /*max_chunk_length=*/2,
                    /*num_keep_alive_refs=*/2, /*delta_encode=*/false,
                    /*disable_compression=*/true);

    // Take two steps (this would finalize a chunk in a regular chunker)
    std::weak_ptr<CellRef> first;
    auto first_want = MakeConstantTensor<tensorflow::DT_FLOAT>({3, 3}, 1);
    REVERB_ASSERT_OK(chunker->Append(first_want, {1, 0}, &first));

    std::weak_ptr<CellRef> second;
    auto second_want = MakeConstantTensor<tensorflow::DT_FLOAT>({3, 3}, 2);
    REVERB_ASSERT_OK(chunker->Append(second_want, {1, 1}, &second));

    // None of the steps should be finalized.
    EXPECT_FALSE(first.lock()->IsReady());
    EXPECT_FALSE(second.lock()->IsReady());

    // We can get the data from the buffer
    tensorflow::Tensor first_got;
    REVERB_ASSERT_OK(first.lock()->GetData(&first_got));
    test::ExpectTensorEqual<float>(first_got, AddBatchDimension(first_want));

    tensorflow::Tensor second_got;
    REVERB_ASSERT_OK(second.lock()->GetData(&second_got));
    test::ExpectTensorEqual<float>(second_got, AddBatchDimension(second_want));
}

TEST(Chunker, AppendValidatesSpecDtype) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/2,
                             /*num_keep_alive_refs=*/5);

  std::weak_ptr<CellRef> ref;
  auto status = chunker->Append(
      MakeZeroTensor<tensorflow::DT_FLOAT>(kFloatSpec), {1, 0}, &ref);

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  absl::StrCat("Tensor of wrong dtype provided for column 0. "
                               "Got float but expected ",
                               Int32Str(), ".")));
}

TEST(Chunker, AppendValidatesSpecShape) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/2,
                             /*num_keep_alive_refs=*/5);

  std::weak_ptr<CellRef> ref;
  auto status =
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(internal::TensorSpec{
                          kIntSpec.name, kIntSpec.dtype, {2}}),
                      {/*episode_id=*/1, /*step=*/0}, &ref);

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Tensor of incompatible shape provided for column 0. "
                  "Got [2] which is incompatible with [1]."));
}

TEST(Chunker, AppendFlushesOnMaxChunkLength) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/2,
                             /*num_keep_alive_refs=*/5);

  // Buffer is not full after first step.
  std::weak_ptr<CellRef> first;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/0}, &first));
  EXPECT_FALSE(first.lock()->IsReady());

  // Second step should trigger flushing of buffer.
  std::weak_ptr<CellRef> second;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/1}, &second));
  EXPECT_TRUE(first.lock()->IsReady());
  EXPECT_TRUE(second.lock()->IsReady());
}

TEST(Chunker, Flush) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/2,
                             /*num_keep_alive_refs=*/5);
  std::weak_ptr<CellRef> ref;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/0}, &ref));
  EXPECT_FALSE(ref.lock()->IsReady());
  REVERB_ASSERT_OK(chunker->Flush());
  EXPECT_TRUE(ref.lock()->IsReady());
}

TEST(Chunker, FlushOnUncompressedDataFails) {
    auto chunker =
        MakeChunker(kIntSpec,
                    /*max_chunk_length=*/2,
                    /*num_keep_alive_refs=*/2, /*delta_encode=*/false,
                    /*disable_compression=*/true);

  absl::Status status = chunker->Flush();
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "FlushLocked is not used when compression is disabled."));
}


TEST(Chunker, DataTensorsLen) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/2,
                             /*num_keep_alive_refs=*/5);
  std::weak_ptr<CellRef> ref;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/0}, &ref));
  EXPECT_FALSE(ref.lock()->IsReady());
  REVERB_ASSERT_OK(chunker->Flush());
  auto cell_ref = ref.lock();
  EXPECT_TRUE(cell_ref->IsReady());
  EXPECT_EQ(cell_ref->GetChunk()->get()->data_tensors_len(),
            cell_ref->GetChunk()->get()->data().tensors_size());
}

TEST(Chunker, ChunkHasBatchDim) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/2,
                             /*num_keep_alive_refs=*/5);

  // Add two data items to trigger the finalization.
  std::weak_ptr<CellRef> ref;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/0}, &ref));
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/1}, &ref));
  ASSERT_TRUE(ref.lock()->IsReady());
  EXPECT_THAT(ref.lock()->GetChunk()->get()->data().tensors(0).tensor_shape(),
              testing::EqualsProto("dim { size: 2} dim { size: 1}"));

  // The batch dim is added even if it only contains a single step.
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/0}, &ref));
  REVERB_ASSERT_OK(chunker->Flush());
  ASSERT_TRUE(ref.lock()->IsReady());
  EXPECT_THAT(ref.lock()->GetChunk()->get()->data().tensors(0).tensor_shape(),
              testing::EqualsProto("dim { size: 1} dim { size: 1}"));
}

TEST(Chunker, DeletesRefsWhenMageAgeExceeded) {
  auto chunker_compressed = MakeChunker(kIntSpec, /*max_chunk_length=*/2,
                             /*num_keep_alive_refs=*/3);
  auto chunker_not_compressed =
        MakeChunker(kIntSpec,
                    /*max_chunk_length=*/2,
                    /*num_keep_alive_refs=*/3, /*delta_encode=*/false,
                    /*disable_compression=*/true);
  for (auto &chunker : {chunker_compressed, chunker_not_compressed}){
    std::weak_ptr<CellRef> first;
    REVERB_ASSERT_OK(
        chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                        {/*episode_id=*/1, /*step=*/0}, &first));
    EXPECT_FALSE(first.expired());

    std::weak_ptr<CellRef> second;
    REVERB_ASSERT_OK(
        chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                        {/*episode_id=*/1, /*step=*/1}, &second));
    EXPECT_FALSE(first.expired());
    EXPECT_FALSE(second.expired());

    std::weak_ptr<CellRef> third;
    REVERB_ASSERT_OK(
        chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                        {/*episode_id=*/1, /*step=*/2}, &third));
    EXPECT_FALSE(first.expired());
    EXPECT_FALSE(second.expired());
    EXPECT_FALSE(third.expired());

    std::weak_ptr<CellRef> fourth;
    REVERB_ASSERT_OK(
        chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                        {/*episode_id=*/1, /*step=*/3}, &fourth));
    EXPECT_TRUE(first.expired());
    EXPECT_FALSE(second.expired());
    EXPECT_FALSE(third.expired());
    EXPECT_FALSE(fourth.expired());
  }
}



TEST(Chunker, GetKeepKeys) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/2,
                             /*num_keep_alive_refs=*/2);

  std::weak_ptr<CellRef> first;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/0}, &first));
  EXPECT_THAT(chunker->GetKeepKeys(), ElementsAre(first.lock()->chunk_key()));

  // The second ref will belong to the same chunk.
  std::weak_ptr<CellRef> second;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/1}, &second));
  EXPECT_THAT(chunker->GetKeepKeys(), ElementsAre(first.lock()->chunk_key()));

  // The third ref will belong to a new chunk. The first ref is now expired but
  // since the second ref belong to the same chunk we expect the chunker to tell
  // us to keep both chunks around.
  std::weak_ptr<CellRef> third;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/2}, &third));
  EXPECT_THAT(chunker->GetKeepKeys(), ElementsAre(second.lock()->chunk_key(),
                                                  third.lock()->chunk_key()));

  // Adding a fourth value results in the second one expiring. The only chunk
  // which should be kept thus is the one referenced by the third and fourth.
  std::weak_ptr<CellRef> fourth;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/3}, &fourth));
  EXPECT_THAT(chunker->GetKeepKeys(), ElementsAre(third.lock()->chunk_key()));
}

TEST(Chunker, ResetClearsRefs) {
  auto chunker_compressed = MakeChunker(kIntSpec, /*max_chunk_length=*/2,
                                        /*num_keep_alive_refs=*/2);
  auto chunker_not_compressed =
      MakeChunker(kIntSpec,
                  /*max_chunk_length=*/2,
                  /*num_keep_alive_refs=*/2, /*delta_encode=*/false,
                  /*disable_compression=*/true);
  for (auto& chunker : {chunker_compressed, chunker_not_compressed}) {
    std::weak_ptr<CellRef> first;
    REVERB_ASSERT_OK(
        chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                        {/*episode_id=*/1, /*step=*/0}, &first));
    std::weak_ptr<CellRef> second;
    REVERB_ASSERT_OK(
        chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                        {/*episode_id=*/1, /*step=*/1}, &second));

    // Before resetting both references are alive.
    EXPECT_FALSE(first.expired());
    EXPECT_FALSE(second.expired());

    // After resetting both references are dead.
    chunker->Reset();
    EXPECT_TRUE(first.expired());
    EXPECT_TRUE(second.expired());
  }
}

TEST(Chunker, ResetRefreshesChunkKey) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/2,
                             /*num_keep_alive_refs=*/2);

  std::weak_ptr<CellRef> first;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/0}, &first));

  // Extract key since the `CellRef` will expire when we reset the
  // `Chunker`.
  uint64_t first_chunk_key = first.lock()->chunk_key();

  chunker->Reset();

  // Take a second step now that the Chunker have been reseted. Note that since
  // `max_chunk_length` hasn't been reached we would expect the second step to
  // be part of the same chunk if `Reset` wasn't called in between.
  std::weak_ptr<CellRef> second;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/1}, &second));

  EXPECT_NE(second.lock()->chunk_key(), first_chunk_key);
}

TEST(Chunker, ResetRefreshesOffset) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/2,
                             /*num_keep_alive_refs=*/2);

  std::weak_ptr<CellRef> first;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/0}, &first));

  chunker->Reset();

  // Take a second step now that the Chunker have been reseted. Note that since
  // `max_chunk_length` hasn't been reached we would expect the second step to
  // be part of the same chunk if `Reset` wasn't called in between.
  std::weak_ptr<CellRef> second;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/1}, &second));

  EXPECT_EQ(second.lock()->offset(), 0);
}

TEST(Chunker, AppendRequiresSameEpisode) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/3,
                             /*num_keep_alive_refs=*/3);

  // Add two steps referencing two different episodes.
  std::weak_ptr<CellRef> first;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/0}, &first));
  std::weak_ptr<CellRef> second;
  auto status = chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                                {/*episode_id=*/2, /*step=*/0}, &second);

  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr(
          "Chunker::Append called with new episode when buffer non empty."));
}

TEST(Chunker, AppendRequiresEpisodeStepIncreases) {
  auto chunker_compressed = MakeChunker(kIntSpec, /*max_chunk_length=*/3,
                                        /*num_keep_alive_refs=*/3);
  auto chunker_not_compressed =
      MakeChunker(kIntSpec,
                  /*max_chunk_length=*/3,
                  /*num_keep_alive_refs=*/3, /*delta_encode=*/false,
                  /*disable_compression=*/true);
  for (auto& chunker : {chunker_compressed, chunker_not_compressed}) {
    // Add two steps referencing two different episodes.
    std::weak_ptr<CellRef> first;
    REVERB_ASSERT_OK(
        chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                        {/*episode_id=*/1, /*step=*/5}, &first));

    // Same step index.
    std::weak_ptr<CellRef> eq;
    auto eq_status =
        chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                        {/*episode_id=*/1, /*step=*/5}, &eq);

    EXPECT_EQ(eq_status.code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_THAT(
        std::string(eq_status.message()),
        ::testing::HasSubstr("Chunker::Append called with an episode step "
                             "which was not greater than already observed."));

    // Smaller step index.
    std::weak_ptr<CellRef> lt;
    auto lt_status =
        chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                        {/*episode_id=*/1, /*step=*/3}, &lt);

    EXPECT_EQ(lt_status.code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_THAT(
        std::string(lt_status.message()),
        ::testing::HasSubstr("Chunker::Append called with an episode step "
                             "which was not greater than already observed."));
  }
}

TEST(Chunker, NonSparseEpisodeRange) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/5,
                             /*num_keep_alive_refs=*/5);

  // Append five consecutive steps.
  std::weak_ptr<CellRef> step;
  for (int i = 0; i < 5; i++) {
    REVERB_ASSERT_OK(
        chunker->Append(MakeConstantTensor<tensorflow::DT_INT32>({1}, 0),
                        {/*episode_id=*/1, /*step=*/i}, &step));
  }

  // Check that the range is non sparse.
  ASSERT_FALSE(step.expired());
  ASSERT_TRUE(step.lock()->IsReady());
  EXPECT_THAT(step.lock()->GetChunk()->get()->sequence_range(),
              testing::EqualsProto("episode_id: 1 start: 0 end: 4"));
}

TEST(Chunker, SparseEpisodeRange) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/5,
                             /*num_keep_alive_refs=*/5);

  // Append five steps with a stride of 2.
  std::weak_ptr<CellRef> step;
  for (int i = 0; i < 5; i++) {
    REVERB_ASSERT_OK(
        chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                        {/*episode_id=*/33, /*step=*/i * 2}, &step));
  }

  // Check that the range is non sparse.
  ASSERT_FALSE(step.expired());
  ASSERT_TRUE(step.lock()->IsReady());
  EXPECT_THAT(
      step.lock()->GetChunk()->get()->sequence_range(),
      testing::EqualsProto("episode_id: 33 start: 0 end: 8 sparse: true"));
}

TEST(Chunker, ApplyConfigChangesMaxChunkLength) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/5,
                             /*num_keep_alive_refs=*/5);

  // Reconfigure the chunk_length to be 1 instead of 5.
  REVERB_ASSERT_OK(
      chunker->ApplyConfig(std::make_shared<ConstantChunkerOptions>(
          /*max_chunk_length=*/1, /*num_keep_alive_refs=*/5)));

  // Appending should now result in chunks being created with each step.
  std::weak_ptr<CellRef> step;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/0}, &step));
  ASSERT_FALSE(step.expired());
  ASSERT_TRUE(step.lock()->IsReady());
  EXPECT_THAT(step.lock()->GetChunk()->get()->sequence_range(),
              testing::EqualsProto("episode_id: 1 start: 0 end: 0"));
}

TEST(Chunker, ApplyConfigChangesNumKeepAliveRefs) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/1,
                             /*num_keep_alive_refs=*/1);

  // Reconfigure num_keep_alive_refs to be 2 instead of 1.
  REVERB_ASSERT_OK(
      chunker->ApplyConfig(std::make_shared<ConstantChunkerOptions>(
          /*max_chunk_length=*/1, /*num_keep_alive_refs=*/2)));

  // The last two steps should now be alive instead of only the last one.
  std::weak_ptr<CellRef> first;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/0}, &first));
  ASSERT_FALSE(first.expired());

  std::weak_ptr<CellRef> second;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/1}, &second));
  ASSERT_FALSE(first.expired());
  ASSERT_FALSE(second.expired());

  std::weak_ptr<CellRef> third;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/2}, &third));
  ASSERT_TRUE(first.expired());
  ASSERT_FALSE(second.expired());
  ASSERT_FALSE(third.expired());
}

TEST(Chunker, ApplyConfigRequireBufferToBeEmpty) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/5,
                             /*num_keep_alive_refs=*/5);

  // Append a step which is not finalized since max_chunk_length is 2.
  std::weak_ptr<CellRef> step;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/0}, &step));

  auto status = chunker->ApplyConfig(std::make_shared<ConstantChunkerOptions>(
      /*max_chunk_length=*/1, /*num_keep_alive_refs=*/5));
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("Flush must be called before ApplyConfig."));

  // Flushing and retrying the same configure call should succeed.
  REVERB_ASSERT_OK(chunker->Flush());
  REVERB_EXPECT_OK(
      chunker->ApplyConfig(std::make_shared<ConstantChunkerOptions>(
          /*max_chunk_length=*/1, /*num_keep_alive_refs=*/5)));
}

TEST(Chunker, OnItemFinalizedForwardsItemAndRefsToOptions) {
  auto options_a = std::make_shared<MockChunkerOptions>();
  EXPECT_CALL(*options_a, GetMaxChunkLength()).WillRepeatedly(Return(1));
  EXPECT_CALL(*options_a, GetNumKeepAliveRefs()).WillRepeatedly(Return(1));

  auto options_b = std::make_shared<MockChunkerOptions>();
  EXPECT_CALL(*options_b, GetMaxChunkLength()).WillRepeatedly(Return(1));
  EXPECT_CALL(*options_b, GetNumKeepAliveRefs()).WillRepeatedly(Return(1));

  auto chunker_a = std::make_shared<Chunker>(kIntSpec, options_a);
  auto chunker_b = std::make_shared<Chunker>(kIntSpec, options_b);

  // Take a step with both chunkers.
  std::weak_ptr<CellRef> ref_a;
  REVERB_ASSERT_OK(
      chunker_a->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                        {/*episode_id=*/1, /*step=*/0}, &ref_a));
  std::weak_ptr<CellRef> ref_b;
  REVERB_ASSERT_OK(
      chunker_b->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                        {/*episode_id=*/1, /*step=*/0}, &ref_b));

  // The call should filter down the refs to only include the refs created by
  // the chunker.
  auto item = testing::MakePrioritizedItem(1, 1.0,
                                           {
                                               *ref_a.lock()->GetChunk()->get(),
                                               *ref_b.lock()->GetChunk()->get(),
                                           });

  EXPECT_CALL(*options_a, OnItemFinalized(testing::EqualsProto(item),
                                          ElementsAre(ref_a.lock())))
      .Times(1);
  REVERB_EXPECT_OK(chunker_a->OnItemFinalized(item, {ref_a.lock()}));

  EXPECT_CALL(*options_b, OnItemFinalized(testing::EqualsProto(item),
                                          ElementsAre(ref_b.lock())))
      .Times(1);
  REVERB_EXPECT_OK(chunker_b->OnItemFinalized(item, {ref_b.lock()}));
}

TEST(Chunker, ApplyConfigRejectsInvalidOptions) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/5,
                             /*num_keep_alive_refs=*/5);
  std::vector<std::pair<int, int>> invalid_options = {
      {0, 5},   // max_chunk_length must be > 0.
      {-1, 5},  // max_chunk_length must be > 0.
      {5, 0},   // num_keep_alive_refs must be > 0.
      {5, -1},  // num_keep_alive_refs must be > 0.
      {6, 5},   // num_keep_alive_refs must be >= max_chunk_length.
  };
  for (const auto [max_chunk_length, num_keep_alive_refs] : invalid_options) {
    auto options = std::make_shared<ConstantChunkerOptions>(
        max_chunk_length, num_keep_alive_refs);
    EXPECT_EQ(chunker->ApplyConfig(options).code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(Chunker, DeltaEncodeIsRespected) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/2,
                             /*num_keep_alive_refs=*/2,
                             /*delta_encode=*/true);

  std::weak_ptr<CellRef> step;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/0}, &step));
  REVERB_ASSERT_OK(chunker->Flush());
  EXPECT_TRUE(step.lock()->GetChunk()->get()->delta_encoded());
}

TEST(Chunker, DataUncompressedSizeIsPopulated) {
  auto chunker = MakeChunker(kIntSpec, /*max_chunk_length=*/2,
                             /*num_keep_alive_refs=*/2,
                             /*delta_encode=*/true);

  std::weak_ptr<CellRef> step;
  REVERB_ASSERT_OK(
      chunker->Append(MakeZeroTensor<tensorflow::DT_INT32>(kIntSpec),
                      {/*episode_id=*/1, /*step=*/0}, &step));
  REVERB_ASSERT_OK(chunker->Flush());
  EXPECT_GT(step.lock()->GetChunk()->get()->data_uncompressed_size(), 0);
}

TEST(ValidateChunkerOptions, Valid) {
  auto options =
      std::make_unique<ConstantChunkerOptions>(/*max_chunk_length=*/2,
                                               /*num_keep_alive_refs=*/2);
  REVERB_EXPECT_OK(ValidateChunkerOptions(options.get()));
}

TEST(ValidateChunkerOptions, ZeroMaxChunkLength) {
  auto options = std::make_unique<ConstantChunkerOptions>(
      /*max_chunk_length=*/0, /*num_keep_alive_refs=*/2);
  auto status = ValidateChunkerOptions(options.get());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("max_chunk_length must be > 0 but got 0."));
}

TEST(ValidateChunkerOptions, NegativeMaxChunkLength) {
  auto options =
      std::make_unique<ConstantChunkerOptions>(/*max_chunk_length=*/-1,
                                               /*num_keep_alive_refs=*/2);
  auto status = ValidateChunkerOptions(options.get());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("max_chunk_length must be > 0 but got -1."));
}

TEST(ValidateChunkerOptions, ZeroNumKeepAliveRefs) {
  auto options = std::make_unique<ConstantChunkerOptions>(
      /*max_chunk_length=*/2, /*num_keep_alive_refs=*/0);
  auto status = ValidateChunkerOptions(options.get());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr("num_keep_alive_refs must be > 0 but got 0."));
}

TEST(ValidateChunkerOptions, NegativeNumKeepAliveRefs) {
  auto options = std::make_unique<ConstantChunkerOptions>(
      /*max_chunk_length=*/2,
      /*num_keep_alive_refs=*/-1);
  auto status = ValidateChunkerOptions(options.get());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr("num_keep_alive_refs must be > 0 but got -1."));
}

TEST(ValidateChunkerOptions, NumKeepAliveLtMaxChunkLength) {
  auto options = std::make_unique<ConstantChunkerOptions>(
      /*max_chunk_length=*/6, /*num_keep_alive_refs=*/5);
  auto status = ValidateChunkerOptions(options.get());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "num_keep_alive_refs (5) must be >= max_chunk_length (6)."));
}

TEST(AutoTunedChunkerOptions, SingleStepItemsAndRandomData) {
  auto options = std::make_shared<AutoTunedChunkerOptions>(10);
  auto chunker = std::make_shared<Chunker>(kLargeFloatSpec, options);

  tensorflow::TensorShape shape;
  ASSERT_TRUE(kLargeFloatSpec.shape.AsTensorShape(&shape));

  // If the data is random and items only take single step then we expect
  // the chunk length to be 1 eventually.
  for (int i = 0; i < 1000; i++) {
    std::weak_ptr<CellRef> ref;
    REVERB_EXPECT_OK(
        chunker->Append(MakeRandomTensor<tensorflow::DT_FLOAT>(shape, 0, 1),
                        {/*episode_id=*/1, /*step=*/i}, &ref));
    if (ref.lock()->IsReady()) {
      REVERB_EXPECT_OK(
          chunker->OnItemFinalized(PrioritizedItem(), {ref.lock()}));
    }
  }

  EXPECT_EQ(options->GetMaxChunkLength(), 1);
}

TEST(AutoTunedChunkerOptions, MultiOverlapStepItemsAndRandomData) {
  auto options = std::make_shared<AutoTunedChunkerOptions>(10);
  auto chunker = std::make_shared<Chunker>(kLargeFloatSpec, options);

  tensorflow::TensorShape shape;
  ASSERT_TRUE(kLargeFloatSpec.shape.AsTensorShape(&shape));

  // If the data is random and items overlap then we expect the chunk length to
  // be small, i.e <= 3. The reason that it isn't necessarily 1 is that the
  // overhead of the proto outweights makes it optimal to have chunk length 2 so
  // the final value will circle between 1 and 3.
  std::deque<std::shared_ptr<CellRef>> last_10_refs;

  for (int i = 0; i < 1000; i++) {
    std::weak_ptr<CellRef> ref;
    REVERB_EXPECT_OK(
        chunker->Append(MakeRandomTensor<tensorflow::DT_FLOAT>(shape, 0, 1),
                        {/*episode_id=*/1, /*step=*/i}, &ref));
    last_10_refs.push_back(ref.lock());
    if (last_10_refs.size() > 10) {
      last_10_refs.pop_front();
    }

    if (std::all_of(last_10_refs.begin(), last_10_refs.end(),
                    [](const auto& r) { return r->IsReady(); })) {
      REVERB_EXPECT_OK(chunker->OnItemFinalized(
          PrioritizedItem(), std::vector<std::shared_ptr<CellRef>>(
                                 last_10_refs.begin(), last_10_refs.end())));
    }
  }

  EXPECT_LE(options->GetMaxChunkLength(), 3);
}

TEST(AutoTunedChunkerOptions, ConstantNonOverlappingItems) {
  auto options = std::make_shared<AutoTunedChunkerOptions>(10);
  auto chunker = std::make_shared<Chunker>(kLargeFloatSpec, options);

  tensorflow::TensorShape shape;
  ASSERT_TRUE(kLargeFloatSpec.shape.AsTensorShape(&shape));

  // When the data doesn't change then compression is king. The chunk length
  // should therefore grow to the max value.
  std::deque<std::shared_ptr<CellRef>> last_10_refs;

  for (int i = 0; i < 1000; i++) {
    std::weak_ptr<CellRef> ref;
    REVERB_EXPECT_OK(
        chunker->Append(MakeConstantTensor<tensorflow::DT_FLOAT>(shape, 33),
                        {/*episode_id=*/1, /*step=*/i}, &ref));
    last_10_refs.push_back(ref.lock());
    if (last_10_refs.size() > 10) {
      last_10_refs.pop_front();
    }

    if (std::all_of(last_10_refs.begin(), last_10_refs.end(),
                    [](const auto& r) { return r->IsReady(); })) {
      REVERB_EXPECT_OK(chunker->OnItemFinalized(
          PrioritizedItem(), std::vector<std::shared_ptr<CellRef>>(
                                 last_10_refs.begin(), last_10_refs.end())));
      last_10_refs.clear();
    }
  }

  EXPECT_EQ(options->GetMaxChunkLength(), 10);
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
