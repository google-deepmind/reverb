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

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cstdint>
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/selectors/heap.h"
#include "reverb/cc/selectors/interface.h"
#include "reverb/cc/selectors/lifo.h"
#include "reverb/cc/selectors/prioritized.h"
#include "reverb/cc/selectors/uniform.h"
#include "reverb/cc/support/tf_util.h"
#include "reverb/cc/support/trajectory_util.h"
#include "reverb/cc/table.h"
#include "reverb/cc/table_extensions/interface.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

namespace deepmind {
namespace reverb {
namespace {

constexpr char kTablesFileName[] = "tables.tfrecord";
constexpr char kChunksFileName[] = "chunks.tfrecord";
constexpr char kDoneFileName[] = "DONE";

using RecordWriterUniquePtr =
    std::unique_ptr<tensorflow::io::RecordWriter,
                    std::function<void(tensorflow::io::RecordWriter*)>>;
using RecordReaderUniquePtr =
    std::unique_ptr<tensorflow::io::RecordReader,
                    std::function<void(tensorflow::io::RecordReader*)>>;

absl::Status OpenWriter(const std::string& path,
                        RecordWriterUniquePtr* writer) {
  std::unique_ptr<tensorflow::WritableFile> file;
  REVERB_RETURN_IF_ERROR(FromTensorflowStatus(
      tensorflow::Env::Default()->NewWritableFile(path, &file)));
  auto* file_ptr = file.release();
  *writer = RecordWriterUniquePtr(new tensorflow::io::RecordWriter(file_ptr),
                                  [file_ptr](tensorflow::io::RecordWriter* w) {
                                    delete w;
                                    delete file_ptr;
                                  });
  return absl::OkStatus();
}

absl::Status OpenReader(const std::string& path,
                        RecordReaderUniquePtr* reader) {
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  REVERB_RETURN_IF_ERROR(FromTensorflowStatus(
      tensorflow::Env::Default()->NewRandomAccessFile(path, &file)));
  auto* file_ptr = file.release();
  *reader = RecordReaderUniquePtr(new tensorflow::io::RecordReader(file_ptr),
                                  [file_ptr](tensorflow::io::RecordReader* r) {
                                    delete r;
                                    delete file_ptr;
                                  });
  return absl::OkStatus();
}

inline absl::Status WriteDone(const std::string& path) {
  std::unique_ptr<tensorflow::WritableFile> file;
  REVERB_RETURN_IF_ERROR(
      FromTensorflowStatus(tensorflow::Env::Default()->NewWritableFile(
          tensorflow::io::JoinPath(path, kDoneFileName), &file)));
  return FromTensorflowStatus(file->Close());
}

inline bool HasDone(const std::string& path) {
  return tensorflow::Env::Default()
      ->FileExists(tensorflow::io::JoinPath(path, kDoneFileName))
      .ok();
}

std::unique_ptr<ItemSelector> MakeDistribution(
    const KeyDistributionOptions& options) {
  switch (options.distribution_case()) {
    case KeyDistributionOptions::kFifo:
      return absl::make_unique<FifoSelector>();
    case KeyDistributionOptions::kLifo:
      return absl::make_unique<LifoSelector>();
    case KeyDistributionOptions::kUniform:
      return absl::make_unique<UniformSelector>();
    case KeyDistributionOptions::kPrioritized:
      return absl::make_unique<PrioritizedSelector>(
          options.prioritized().priority_exponent());
    case KeyDistributionOptions::kHeap:
      return absl::make_unique<HeapSelector>(options.heap().min_heap());
    case KeyDistributionOptions::DISTRIBUTION_NOT_SET:
      REVERB_LOG(REVERB_FATAL) << "Selector not set";
    default:
      REVERB_LOG(REVERB_FATAL) << "Selector not supported";
  }
}

inline size_t find_table_index(
    const std::vector<std::shared_ptr<Table>>* tables,
    const std::string& name) {
  for (int i = 0; i < tables->size(); i++) {
    if (tables->at(i)->name() == name) return i;
  }
  return -1;
}

}  // namespace

TFRecordCheckpointer::TFRecordCheckpointer(
    std::string root_dir, std::string group,
    absl::optional<std::string> fallback_checkpoint_path)
    : root_dir_(std::move(root_dir)),
      group_(std::move(group)),
      fallback_checkpoint_path_(std::move(fallback_checkpoint_path)) {
  REVERB_LOG(REVERB_INFO) << " Initializing TFRecordCheckpointer in "
                          << root_dir_
                          << (fallback_checkpoint_path_.has_value()
                                  ? absl::StrCat(
                                        " and fallback directory ",
                                        fallback_checkpoint_path_.value(), ".")
                                  : ".");
}

absl::Status TFRecordCheckpointer::Save(std::vector<Table*> tables,
                                        int keep_latest, std::string* path) {
  if (keep_latest <= 0) {
    return absl::InvalidArgumentError(
        "TFRecordCheckpointer must have keep_latest > 0.");
  }
  if (!group_.empty()) {
    return absl::InvalidArgumentError(
        "Setting non-empty group is not supported");
  }

  std::string dir_path =
      tensorflow::io::JoinPath(root_dir_, absl::FormatTime(absl::Now()));
  REVERB_RETURN_IF_ERROR(FromTensorflowStatus(
      tensorflow::Env::Default()->RecursivelyCreateDir(dir_path)));

  RecordWriterUniquePtr table_writer;
  REVERB_RETURN_IF_ERROR(OpenWriter(
      tensorflow::io::JoinPath(dir_path, kTablesFileName), &table_writer));

  absl::flat_hash_set<std::shared_ptr<ChunkStore::Chunk>> chunks;
  for (Table* table : tables) {
    auto checkpoint = table->Checkpoint();
    chunks.merge(checkpoint.chunks);
    REVERB_RETURN_IF_ERROR(FromTensorflowStatus(
        table_writer->WriteRecord(checkpoint.checkpoint.SerializeAsString())));
  }

  REVERB_RETURN_IF_ERROR(FromTensorflowStatus(table_writer->Close()));
  table_writer = nullptr;

  RecordWriterUniquePtr chunk_writer;
  REVERB_RETURN_IF_ERROR(OpenWriter(
      tensorflow::io::JoinPath(dir_path, kChunksFileName), &chunk_writer));

  for (const auto& chunk : chunks) {
    REVERB_RETURN_IF_ERROR(FromTensorflowStatus(
        chunk_writer->WriteRecord(chunk->data().SerializeAsString())));
  }
  REVERB_RETURN_IF_ERROR(FromTensorflowStatus(chunk_writer->Close()));
  chunk_writer = nullptr;

  // Both chunks and table checkpoint has now been written so we can proceed to
  // add the DONE-file.
  REVERB_RETURN_IF_ERROR(WriteDone(dir_path));

  // Delete the older checkpoints.
  std::vector<std::string> filenames;
  REVERB_RETURN_IF_ERROR(
      FromTensorflowStatus(tensorflow::Env::Default()->GetMatchingPaths(
          tensorflow::io::JoinPath(root_dir_, "*"), &filenames)));
  std::sort(filenames.begin(), filenames.end());
  int history_counter = 0;
  for (auto it = filenames.rbegin(); it != filenames.rend(); it++) {
    if (++history_counter > keep_latest) {
      tensorflow::int64 undeleted_files;
      tensorflow::int64 undeleted_dirs;
      REVERB_RETURN_IF_ERROR(
          FromTensorflowStatus(tensorflow::Env::Default()->DeleteRecursively(
              *it, &undeleted_files, &undeleted_dirs)));
    }
  }

  *path = std::move(dir_path);
  return absl::OkStatus();
}

absl::Status TFRecordCheckpointer::Load(
    absl::string_view path, ChunkStore* chunk_store,
    std::vector<std::shared_ptr<Table>>* tables) {
  REVERB_LOG(REVERB_INFO) << "Loading checkpoint from " << std::string(path);
  if (!HasDone(std::string(path))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Load called with invalid checkpoint path: ", std::string(path)));
  }
  // Insert data first to ensure that all data referenced by the tables
  // exists. Keep the map of chunks around so that none of the chunks are
  // cleaned up before all the tables have been loaded.
  internal::flat_hash_map<ChunkStore::Key, std::shared_ptr<ChunkStore::Chunk>>
      chunk_by_key;
  {
    RecordReaderUniquePtr chunk_reader;
    REVERB_RETURN_IF_ERROR(
        OpenReader(tensorflow::io::JoinPath(std::string(path), kChunksFileName),
                   &chunk_reader));

    ChunkData chunk_data;
    absl::Status chunk_status;
    tensorflow::uint64 chunk_offset = 0;
    tensorflow::tstring chunk_record;
    do {
      chunk_status = FromTensorflowStatus(
          chunk_reader->ReadRecord(&chunk_offset, &chunk_record));
      if (!chunk_status.ok()) break;
      if (!chunk_data.ParseFromArray(chunk_record.data(),
                                     chunk_record.size())) {
        return absl::DataLossError(
            absl::StrCat("Could not parse TFRecord as ChunkData: '",
                         absl::string_view(chunk_record), "'"));
      }
      if (chunk_data.deprecated_data_size()) {
        if (!chunk_data.data().tensors().empty()) {
          return absl::InternalError(
              absl::StrCat("Checkpoint ChunkData at offset: ", chunk_offset,
              " has both data and deprecated_data."));
        }
        chunk_data.mutable_data()->mutable_tensors()->Swap(
            chunk_data.mutable_deprecated_data());
      }
      chunk_by_key[chunk_data.chunk_key()] = chunk_store->Insert(chunk_data);
    } while (chunk_status.ok());
    if (!absl::IsOutOfRange(chunk_status)) {
      return chunk_status;
    }
  }

  RecordReaderUniquePtr table_reader;
  REVERB_RETURN_IF_ERROR(
      OpenReader(tensorflow::io::JoinPath(std::string(path), kTablesFileName),
                 &table_reader));

  PriorityTableCheckpoint checkpoint;
  absl::Status table_status;
  tensorflow::uint64 table_offset = 0;
  tensorflow::tstring table_record;
  do {
    table_status = FromTensorflowStatus(
        table_reader->ReadRecord(&table_offset, &table_record));
    if (!table_status.ok()) break;
    if (!checkpoint.ParseFromArray(table_record.data(), table_record.size())) {
      return absl::DataLossError(
          absl::StrCat("Could not parse TFRecord as Checkpoint: '",
                       absl::string_view(table_record), "'"));
    }

    int index = find_table_index(tables, checkpoint.table_name());
    if (index == -1) {
      std::vector<std::string> table_names;
      for (const auto& table : *tables) {
        table_names.push_back(absl::StrCat("'", table->name(), "'"));
      }
      return absl::InvalidArgumentError(absl::StrCat(
          "Trying to load table ", checkpoint.table_name(),
          " but table was not found in provided list of tables. Available "
          "tables: [",
          absl::StrJoin(table_names, ", "), "]"));
    }

    auto sampler = MakeDistribution(checkpoint.sampler());
    auto remover = MakeDistribution(checkpoint.remover());
    auto rate_limiter =
        std::make_shared<RateLimiter>(checkpoint.rate_limiter());
    auto extensions = tables->at(index)->UnsafeClearExtensions();
    auto signature =
        checkpoint.has_signature()
            ? absl::make_optional(std::move(checkpoint.signature()))
            : absl::nullopt;

    auto table = std::make_shared<Table>(
        /*name=*/checkpoint.table_name(),
        /*sampler=*/std::move(sampler),
        /*remover=*/std::move(remover),
        /*max_size=*/checkpoint.max_size(),
        /*max_times_sampled=*/checkpoint.max_times_sampled(),
        /*rate_limiter=*/std::move(rate_limiter),
        /*extensions=*/std::move(extensions),
        /*signature=*/std::move(signature));
    table->set_num_deleted_episodes_from_checkpoint(
        checkpoint.num_deleted_episodes());

    for (const auto& checkpoint_item : checkpoint.items()) {
      Table::Item insert_item;
      insert_item.item = checkpoint_item;

      if (insert_item.item.has_deprecated_sequence_range() &&
          insert_item.item.has_flat_trajectory()) {
        return absl::InternalError(
            absl::StrCat("Item ", insert_item.item.key(),
                         " has both deprecated and new trajectory format: ",
                         insert_item.item.DebugString(), "."));
      }

      if (insert_item.item.has_deprecated_sequence_range()) {
        std::vector<std::shared_ptr<ChunkStore::Chunk>> trajectory_chunks;
        REVERB_CHECK_OK(FromTensorflowStatus(chunk_store->Get(
            insert_item.item.deprecated_chunk_keys(), &trajectory_chunks)));

        *insert_item.item.mutable_flat_trajectory() =
            internal::FlatTimestepTrajectory(
                trajectory_chunks,
                insert_item.item.deprecated_sequence_range().offset(),
                insert_item.item.deprecated_sequence_range().length());

        insert_item.item.clear_deprecated_sequence_range();
        insert_item.item.clear_deprecated_chunk_keys();
      }

      for (const auto& key :
           internal::GetChunkKeys(insert_item.item.flat_trajectory())) {
        REVERB_CHECK(chunk_by_key.contains(key));
        insert_item.chunks.push_back(chunk_by_key[key]);
      }

      // The original table has already been destroyed so if this fails then
      // there is way to recover.
      REVERB_CHECK_OK(table->InsertCheckpointItem(std::move(insert_item)));
    }

    tables->at(index).swap(table);
  } while (table_status.ok());

  if (!absl::IsOutOfRange(table_status)) {
    return table_status;
  }
  return absl::OkStatus();
}

absl::Status TFRecordCheckpointer::LoadLatest(
    ChunkStore* chunk_store, std::vector<std::shared_ptr<Table>>* tables) {
  REVERB_LOG(REVERB_INFO) << "Loading latest checkpoint from " << root_dir_;
  std::vector<std::string> filenames;
  REVERB_RETURN_IF_ERROR(
      FromTensorflowStatus(tensorflow::Env::Default()->GetMatchingPaths(
          tensorflow::io::JoinPath(root_dir_, "*"), &filenames)));
  std::sort(filenames.begin(), filenames.end());
  for (auto it = filenames.rbegin(); it != filenames.rend(); it++) {
    if (HasDone(*it)) {
      return Load(
          tensorflow::io::JoinPath(root_dir_, tensorflow::io::Basename(*it)),
          chunk_store, tables);
    }
  }
  return absl::NotFoundError(
      absl::StrCat("No checkpoint found in ", root_dir_));
}

absl::Status TFRecordCheckpointer::LoadFallbackCheckpoint(
    ChunkStore* chunk_store, std::vector<std::shared_ptr<Table>>* tables) {
  if (!fallback_checkpoint_path_.has_value()) {
    return absl::NotFoundError("No fallback checkpoint path provided.");
  }
  if (HasDone(fallback_checkpoint_path_.value())) {
    return Load(fallback_checkpoint_path_.value(), chunk_store, tables);
  }
  return absl::NotFoundError(absl::StrCat("No checkpoint found in ",
                                          fallback_checkpoint_path_.value()));
}

std::string TFRecordCheckpointer::DebugString() const {
  return absl::StrCat("TFRecordCheckpointer(root_dir=", root_dir_,
                      ", group=", group_, ")");
}

}  // namespace reverb
}  // namespace deepmind
