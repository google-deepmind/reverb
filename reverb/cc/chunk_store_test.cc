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

#include "reverb/cc/chunk_store.h"

#include <atomic>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/testing/proto_test_util.h"

namespace deepmind {
namespace reverb {
namespace {

using ChunkVector = ::std::vector<::std::shared_ptr<ChunkStore::Chunk>>;

TEST(ChunkStoreTest, GetAfterInsertSucceeds) {
  ChunkStore store;
  std::shared_ptr<ChunkStore::Chunk> inserted =
      store.Insert(testing::MakeChunkData(2));
  ChunkVector chunks;
  REVERB_ASSERT_OK(store.Get({2}, &chunks));
  EXPECT_EQ(inserted, chunks[0]);
}

TEST(ChunkStoreTest, GetFailsWhenKeyDoesNotExist) {
  ChunkStore store;
  ChunkVector chunks;
  EXPECT_TRUE(absl::IsNotFound(store.Get({2}, &chunks)));
}

TEST(ChunkStoreTest, GetFailsAfterChunkIsDestroyed) {
  ChunkStore store;
  std::shared_ptr<ChunkStore::Chunk> inserted =
      store.Insert(testing::MakeChunkData(1));
  inserted = nullptr;
  ChunkVector chunks;
  EXPECT_TRUE(absl::IsNotFound(store.Get({2}, &chunks)));
}

TEST(ChunkStoreTest, InsertingTwiceReturnsExistingChunk) {
  ChunkStore store;
  ChunkData data = testing::MakeChunkData(2);
  data.set_data_tensors_len(1);
  data.mutable_data()->add_tensors();
  std::shared_ptr<ChunkStore::Chunk> first =
      store.Insert(testing::MakeChunkData(2));
  EXPECT_NE(first, nullptr);
  std::shared_ptr<ChunkStore::Chunk> second =
      store.Insert(testing::MakeChunkData(2));
  EXPECT_EQ(first, second);
}

TEST(ChunkStoreTest, InsertingTwiceSucceedsWhenChunkIsDestroyed) {
  ChunkStore store;
  std::shared_ptr<ChunkStore::Chunk> first =
      store.Insert(testing::MakeChunkData(1));
  EXPECT_NE(first, nullptr);
  first = nullptr;
  std::shared_ptr<ChunkStore::Chunk> second =
      store.Insert(testing::MakeChunkData(1));
  EXPECT_NE(second, nullptr);
}

TEST(ChunkStoreTest, ConcurrentCalls) {
  ChunkStore store;
  std::vector<std::unique_ptr<internal::Thread>> bundle;
  std::atomic<int> count(0);
  for (ChunkStore::Key i = 0; i < 1000; i++) {
    bundle.push_back(internal::StartThread("", [i, &store, &count] {
      std::shared_ptr<ChunkStore::Chunk> first =
          store.Insert(testing::MakeChunkData(i));
      ChunkVector chunks;
      REVERB_ASSERT_OK(store.Get({i}, &chunks));
      first = nullptr;
      while (!absl::IsNotFound(store.Get({i}, &chunks))) {
      }
      count++;
    }));
  }
  bundle.clear();  // Joins all threads.
  EXPECT_EQ(count, 1000);
}

TEST(ChunkTest, Length) {
  ChunkData data;
  data.mutable_sequence_range()->set_start(5);
  data.mutable_sequence_range()->set_end(10);
  EXPECT_EQ(ChunkStore::Chunk(data).num_rows(), 6);
  data.mutable_sequence_range()->set_end(5);
  EXPECT_EQ(ChunkStore::Chunk(data).num_rows(), 1);
}

TEST(ChunkTest, NumColumns) {
  ChunkData data;
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(ChunkStore::Chunk(data).num_columns(), i);
    data.mutable_data()->add_tensors();
    data.set_data_tensors_len(i+1);
  }
}

TEST(ChunkTest, Key) {
  for (int i = 0; i < 5; i++) {
    ChunkData data;
    data.set_chunk_key(i);
    EXPECT_EQ(ChunkStore::Chunk(data).key(), i);
  }
}

TEST(ChunkTest, EpisodeId) {
  for (int i = 0; i < 5; i++) {
    ChunkData data;
    data.mutable_sequence_range()->set_episode_id(i);
    EXPECT_EQ(ChunkStore::Chunk(data).episode_id(), i);
  }
}

TEST(ChunkTest, UncompressedDataSize_ReturnsOneIfUndefined) {
  EXPECT_EQ(ChunkStore::Chunk(ChunkData{}).uncompressed_data_size(), 1);
}

TEST(ChunkTest, UncompressedDataSize_ReturnsValueIfDefined) {
  ChunkData data;
  data.set_data_uncompressed_size(1337);
  EXPECT_EQ(ChunkStore::Chunk(data).uncompressed_data_size(), 1337);
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
