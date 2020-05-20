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

#include "reverb/cc/platform/tfrecord_checkpointer.h"

#include <cfloat>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/selectors/heap.h"
#include "reverb/cc/selectors/prioritized.h"
#include "reverb/cc/selectors/uniform.h"
#include "reverb/cc/table.h"
#include "reverb/cc/testing/proto_test_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"

namespace deepmind {
namespace reverb {
namespace {

using ::deepmind::reverb::testing::EqualsProto;

inline std::string MakeRoot() {
  std::string name;
  REVERB_CHECK(tensorflow::Env::Default()->LocalTempFilename(&name));
  return name;
}

std::unique_ptr<Table> MakeUniformTable(const std::string& name) {
  return absl::make_unique<Table>(
      name, absl::make_unique<UniformSelector>(),
      absl::make_unique<FifoSelector>(), 1000, 0,
      absl::make_unique<RateLimiter>(1.0, 1, -DBL_MAX, DBL_MAX));
}

std::unique_ptr<Table> MakePrioritizedTable(const std::string& name,
                                            double exponent) {
  return absl::make_unique<Table>(
      name, absl::make_unique<PrioritizedSelector>(exponent),
      absl::make_unique<HeapSelector>(), 1000, 0,
      absl::make_unique<RateLimiter>(1.0, 1, -DBL_MAX, DBL_MAX));
}

TEST(TFRecordCheckpointerTest, CreatesDirectoryInRoot) {
  std::string root = MakeRoot();
  TFRecordCheckpointer checkpointer(root);
  std::string path;
  auto* env = tensorflow::Env::Default();
  TF_ASSERT_OK(checkpointer.Save(std::vector<Table*>{}, 1, &path));
  ASSERT_EQ(tensorflow::io::Dirname(path), root);
  TF_EXPECT_OK(env->FileExists(path));
}

TEST(TFRecordCheckpointerTest, SaveAndLoad) {
  ChunkStore chunk_store;

  std::vector<std::shared_ptr<Table>> tables;
  tables.push_back(MakeUniformTable("uniform"));
  tables.push_back(MakePrioritizedTable("prioritized_a", 0.5));
  tables.push_back(MakePrioritizedTable("prioritized_b", 0.9));

  std::vector<ChunkStore::Key> chunk_keys;
  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < tables.size(); j++) {
      chunk_keys.push_back((j + 1) * 1000 + i);
      auto chunk =
          chunk_store.Insert(testing::MakeChunkData(chunk_keys.back()));
      TF_EXPECT_OK(tables[j]->InsertOrAssign(
          {testing::MakePrioritizedItem(i, i, {chunk->data()}), {chunk}}));
    }
  }

  for (int i = 0; i < 100; i++) {
    for (auto& table : tables) {
      Table::SampledItem sample;
      TF_EXPECT_OK(table->Sample(&sample));
    }
  }

  TFRecordCheckpointer checkpointer(MakeRoot());

  std::string path;
  TF_ASSERT_OK(checkpointer.Save(
      {tables[0].get(), tables[1].get(), tables[2].get()}, 1, &path));

  ChunkStore loaded_chunk_store;
  std::vector<std::shared_ptr<Table>> loaded_tables;
  loaded_tables.push_back(MakeUniformTable("uniform"));
  loaded_tables.push_back(MakePrioritizedTable("prioritized_a", 0.5));
  loaded_tables.push_back(MakePrioritizedTable("prioritized_b", 0.9));
  TF_ASSERT_OK(checkpointer.Load(tensorflow::io::Basename(path),
                                 &loaded_chunk_store, &loaded_tables));

  // Check that all the chunks have been added.
  std::vector<std::shared_ptr<ChunkStore::Chunk>> chunks;
  TF_EXPECT_OK(loaded_chunk_store.Get(chunk_keys, &chunks));

  // Check that the number of items matches for the loaded tables.
  for (int i = 0; i < tables.size(); i++) {
    EXPECT_EQ(loaded_tables[i]->size(), tables[i]->size());
  }

  // Sample a random item and check that it matches the item in the original
  // table.
  for (int i = 0; i < tables.size(); i++) {
    Table::SampledItem sample;
    TF_EXPECT_OK(loaded_tables[i]->Sample(&sample));
    bool item_found = false;
    for (auto& item : tables[i]->Copy()) {
      if (item.item.key() == sample.item.key()) {
        item_found = true;
        item.item.set_times_sampled(item.item.times_sampled() + 1);
        EXPECT_THAT(item.item, EqualsProto(sample.item));
        break;
      }
    }
    EXPECT_TRUE(item_found);
  }
}

TEST(TFRecordCheckpointerTest, SaveDeletesOldData) {
  ChunkStore chunk_store;

  std::vector<std::shared_ptr<Table>> tables;
  tables.push_back(MakeUniformTable("uniform"));
  tables.push_back(MakePrioritizedTable("prioritized_a", 0.5));
  tables.push_back(MakePrioritizedTable("prioritized_b", 0.9));

  std::vector<ChunkStore::Key> chunk_keys;
  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < tables.size(); j++) {
      chunk_keys.push_back((j + 1) * 1000 + i);
      auto chunk =
          chunk_store.Insert(testing::MakeChunkData(chunk_keys.back()));
      TF_EXPECT_OK(tables[j]->InsertOrAssign(
          {testing::MakePrioritizedItem(i, i, {chunk->data()}), {chunk}}));
    }
  }

  for (int i = 0; i < 100; i++) {
    for (auto& table : tables) {
      Table::SampledItem sample;
      TF_EXPECT_OK(table->Sample(&sample));
    }
  }

  auto test = [&tables](int keep_latest) {
    auto root = MakeRoot();
    TFRecordCheckpointer checkpointer(root);

    for (int i = 0; i < 10; i++) {
      std::string path;
      TF_ASSERT_OK(
          checkpointer.Save({tables[0].get(), tables[1].get(), tables[2].get()},
                            keep_latest, &path));

      std::vector<std::string> filenames;
      TF_ASSERT_OK(tensorflow::Env::Default()->GetMatchingPaths(
          tensorflow::io::JoinPath(root, "*"), &filenames));
      ASSERT_EQ(filenames.size(), std::min(keep_latest, i + 1));
    }
  };
  test(1);  // Keep one checkpoint.
  test(3);  // Edge case keep_latest == num_tables
  test(5);  // Edge case keep_latest > num_tables
}

TEST(TFRecordCheckpointerTest, KeepLatestZeroReturnsError) {
  ChunkStore chunk_store;

  std::vector<std::shared_ptr<Table>> tables;
  tables.push_back(MakeUniformTable("uniform"));
  tables.push_back(MakePrioritizedTable("prioritized_a", 0.5));
  tables.push_back(MakePrioritizedTable("prioritized_b", 0.9));

  std::vector<ChunkStore::Key> chunk_keys;
  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < tables.size(); j++) {
      chunk_keys.push_back((j + 1) * 1000 + i);
      auto chunk =
          chunk_store.Insert(testing::MakeChunkData(chunk_keys.back()));
      TF_EXPECT_OK(tables[j]->InsertOrAssign(
          {testing::MakePrioritizedItem(i, i, {chunk->data()}), {chunk}}));
    }
  }

  for (int i = 0; i < 100; i++) {
    for (auto& table : tables) {
      Table::SampledItem sample;
      TF_EXPECT_OK(table->Sample(&sample));
    }
  }

  TFRecordCheckpointer checkpointer(MakeRoot());
  std::string path;
  EXPECT_EQ(
      checkpointer
          .Save({tables[0].get(), tables[1].get(), tables[2].get()}, 0, &path)
          .code(),
      tensorflow::error::INVALID_ARGUMENT);
}

TEST(TFRecordCheckpointerTest, LoadLatestInEmptyDir) {
  TFRecordCheckpointer checkpointer(MakeRoot());
  ChunkStore chunk_store;
  std::vector<std::shared_ptr<Table>> tables;
  EXPECT_EQ(checkpointer.LoadLatest(&chunk_store, &tables).code(),
            tensorflow::error::NOT_FOUND);
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
