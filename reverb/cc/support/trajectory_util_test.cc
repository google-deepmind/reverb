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

#include "reverb/cc/support/trajectory_util.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/tensor_compression.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "reverb/cc/testing/tensor_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace deepmind {
namespace reverb {
namespace internal {
namespace {

using ::testing::ElementsAre;

TEST(GetChunkKeys, DedupsTimestepTrajectory) {
  auto trajectory = FlatTimestepTrajectory(
      /*chunk_keys=*/{1, 2, 3}, /*chunk_lengths=*/{2, 2, 2},
      /*num_columns=*/2, /*offset=*/1, /*length=*/4);
  EXPECT_THAT(GetChunkKeys(trajectory), ElementsAre(1, 2, 3));
}

TEST(GetChunkKeys, DedupsNonTimestepTrajectory) {
  FlatTrajectory trajectory;

  auto* first = trajectory.add_columns();
  first->add_chunk_slices()->set_chunk_key(1);
  first->add_chunk_slices()->set_chunk_key(2);
  first->add_chunk_slices()->set_chunk_key(4);

  auto* second = trajectory.add_columns();
  second->add_chunk_slices()->set_chunk_key(2);
  second->add_chunk_slices()->set_chunk_key(3);

  EXPECT_THAT(GetChunkKeys(trajectory), ElementsAre(1, 2, 4, 3));
}

TEST(FlatTimestepTrajectory, CreateTrajectoryFromChunks) {
  ChunkData first;
  first.set_chunk_key(1);
  first.mutable_sequence_range()->set_start(0);
  first.mutable_sequence_range()->set_end(4);
  first.mutable_data()->add_tensors();
  first.mutable_data()->add_tensors();

  ChunkData second;
  second.set_chunk_key(2);
  second.mutable_sequence_range()->set_start(5);
  second.mutable_sequence_range()->set_end(7);
  second.mutable_data()->add_tensors();
  second.mutable_data()->add_tensors();

  std::vector<std::shared_ptr<ChunkStore::Chunk>> chunks = {
      std::make_shared<ChunkStore::Chunk>(std::move(first)),
      std::make_shared<ChunkStore::Chunk>(std::move(second)),
  };

  auto trajectory = FlatTimestepTrajectory(chunks, 1, 5);
  EXPECT_THAT(trajectory, testing::EqualsProto(R"(
                columns: {
                  chunk_slices: { chunk_key: 1 offset: 1 length: 4 index: 0 }
                  chunk_slices: { chunk_key: 2 offset: 0 length: 1 index: 0 }
                }
                columns: {
                  chunk_slices: { chunk_key: 1 offset: 1 length: 4 index: 1 }
                  chunk_slices: { chunk_key: 2 offset: 0 length: 1 index: 1 }
                }
              )"));
}

TEST(FlatTimestepTrajectory, CreateTrajectoryFromVectors) {
  auto trajectory = FlatTimestepTrajectory(
      /*chunk_keys=*/{1, 2},
      /*chunk_lengths=*/{4, 4}, /*num_columns=*/2, /*offset=*/2, /*length=*/4);
  EXPECT_THAT(trajectory, testing::EqualsProto(R"(
                columns: {
                  chunk_slices: { chunk_key: 1 offset: 2 length: 2 index: 0 }
                  chunk_slices: { chunk_key: 2 offset: 0 length: 2 index: 0 }
                }
                columns: {
                  chunk_slices: { chunk_key: 1 offset: 2 length: 2 index: 1 }
                  chunk_slices: { chunk_key: 2 offset: 0 length: 2 index: 1 }
                }
              )"));
}

TEST(IsTimestepTrajectory, SingleColumn) {
  FlatTrajectory trajectory;
  auto* col = trajectory.add_columns();

  auto* first = col->add_chunk_slices();
  first->set_chunk_key(1);
  first->set_length(3);
  first->set_offset(2);

  auto* second = col->add_chunk_slices();
  second->set_chunk_key(2);
  second->set_length(2);
  second->set_offset(0);

  EXPECT_TRUE(IsTimestepTrajectory(trajectory));
}

TEST(IsTimestepTrajectory, SingleColumnWithGap) {
  FlatTrajectory trajectory;
  auto* col = trajectory.add_columns();

  auto* first = col->add_chunk_slices();
  first->set_chunk_key(1);
  first->set_length(3);
  first->set_offset(2);

  auto* second = col->add_chunk_slices();
  second->set_chunk_key(2);
  second->set_length(2);
  second->set_offset(1);

  EXPECT_FALSE(IsTimestepTrajectory(trajectory));
}

TEST(IsTimestepTrajectory, MultiColumn) {
  FlatTrajectory trajectory;
  for (int i = 0; i < 3; i++) {
    auto* col = trajectory.add_columns();

    auto* first = col->add_chunk_slices();
    first->set_chunk_key(1);
    first->set_length(3);
    first->set_offset(2);
    first->set_index(i);

    auto* second = col->add_chunk_slices();
    second->set_chunk_key(2);
    second->set_length(2);
    second->set_offset(0);
    second->set_index(i);
  }

  EXPECT_TRUE(IsTimestepTrajectory(trajectory));
}

class IsTimestepTrajectoryTest : public ::testing::Test {
 protected:
  IsTimestepTrajectoryTest() {
    for (int i = 0; i < 3; i++) {
      auto* col = valid_.add_columns();

      auto* first = col->add_chunk_slices();
      first->set_chunk_key(1);
      first->set_length(3);
      first->set_offset(2);
      first->set_index(i);

      auto* second = col->add_chunk_slices();
      second->set_chunk_key(2);
      second->set_length(2);
      second->set_offset(0);
      second->set_index(i);
    }
  }

  FlatTrajectory GetValid() const { return valid_; }

 private:
  FlatTrajectory valid_;
};

TEST_F(IsTimestepTrajectoryTest, KeyChanged) {
  auto trajectory = GetValid();
  trajectory.mutable_columns(1)->mutable_chunk_slices(0)->set_chunk_key(3);
  EXPECT_FALSE(IsTimestepTrajectory(trajectory));
}

TEST_F(IsTimestepTrajectoryTest, LengthChanged) {
  auto trajectory = GetValid();
  trajectory.mutable_columns(1)->mutable_chunk_slices(0)->set_length(5);
  EXPECT_FALSE(IsTimestepTrajectory(trajectory));
}

TEST_F(IsTimestepTrajectoryTest, NumSlicesChanged) {
  auto trajectory = GetValid();
  trajectory.mutable_columns(1)->mutable_chunk_slices()->RemoveLast();
  EXPECT_FALSE(IsTimestepTrajectory(trajectory));
}

TEST_F(IsTimestepTrajectoryTest, IndexIsInvalid) {
  auto trajectory = GetValid();
  trajectory.mutable_columns(1)->mutable_chunk_slices(0)->set_index(3);
  EXPECT_FALSE(IsTimestepTrajectory(trajectory));
}

TEST(TimestepTrajectoryLength, AccumulatesSlices) {
  FlatTrajectory trajectory;
  auto* col = trajectory.add_columns();

  auto* first = col->add_chunk_slices();
  first->set_chunk_key(1);
  first->set_length(3);
  first->set_offset(2);

  auto* second = col->add_chunk_slices();
  second->set_chunk_key(2);
  second->set_length(2);
  second->set_offset(0);

  EXPECT_EQ(TimestepTrajectoryLength(trajectory), 5);

  // Changing the offset should not impact it.
  first->set_offset(5);
  EXPECT_EQ(TimestepTrajectoryLength(trajectory), 5);

  // Changing the lengths of slices should impact the total length.
  first->set_length(5);
  EXPECT_EQ(TimestepTrajectoryLength(trajectory), 7);

  second->set_length(1);
  EXPECT_EQ(TimestepTrajectoryLength(trajectory), 6);
}

TEST(UnpackChunkColumn, SelectsCorrectColumn) {
  tensorflow::Tensor first_col_tensor(static_cast<int32_t>(1337));
  tensorflow::Tensor second_col_tensor(static_cast<int32_t>(9000));

  ChunkData data;
  CompressTensorAsProto(first_col_tensor, data.mutable_data()->add_tensors());
  CompressTensorAsProto(second_col_tensor, data.mutable_data()->add_tensors());

  tensorflow::Tensor first_got;
  TF_EXPECT_OK(UnpackChunkColumn(data, 0, &first_got));
  test::ExpectTensorEqual<int32_t>(first_got, first_col_tensor);

  tensorflow::Tensor second_got;
  TF_EXPECT_OK(UnpackChunkColumn(data, 1, &second_got));
  test::ExpectTensorEqual<int32_t>(second_got, second_col_tensor);
}

}  // namespace
}  // namespace internal
}  // namespace reverb
}  // namespace deepmind
