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
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/selectors/heap.h"
#include "reverb/cc/selectors/interface.h"
#include "reverb/cc/selectors/lifo.h"
#include "reverb/cc/selectors/prioritized.h"
#include "reverb/cc/selectors/uniform.h"
#include "reverb/cc/table.h"
#include "reverb/cc/table_extensions/interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
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

tensorflow::Status OpenWriter(const std::string& path,
                              RecordWriterUniquePtr* writer) {
  std::unique_ptr<tensorflow::WritableFile> file;
  TF_RETURN_IF_ERROR(tensorflow::Env::Default()->NewWritableFile(path, &file));
  auto* file_ptr = file.release();
  *writer = RecordWriterUniquePtr(new tensorflow::io::RecordWriter(file_ptr),
                                  [file_ptr](tensorflow::io::RecordWriter* w) {
                                    delete w;
                                    delete file_ptr;
                                  });
  return tensorflow::Status::OK();
}

tensorflow::Status OpenReader(const std::string& path,
                              RecordReaderUniquePtr* reader) {
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(
      tensorflow::Env::Default()->NewRandomAccessFile(path, &file));
  auto* file_ptr = file.release();
  *reader = RecordReaderUniquePtr(new tensorflow::io::RecordReader(file_ptr),
                                  [file_ptr](tensorflow::io::RecordReader* r) {
                                    delete r;
                                    delete file_ptr;
                                  });
  return tensorflow::Status::OK();
}

inline tensorflow::Status WriteDone(const std::string& path) {
  std::unique_ptr<tensorflow::WritableFile> file;
  TF_RETURN_IF_ERROR(tensorflow::Env::Default()->NewWritableFile(
      tensorflow::io::JoinPath(path, kDoneFileName), &file));
  return file->Close();
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

TFRecordCheckpointer::TFRecordCheckpointer(std::string root_dir,
                                           std::string group)
    : root_dir_(std::move(root_dir)), group_(std::move(group)) {
  REVERB_LOG(REVERB_INFO) << "Initializing TFRecordCheckpointer in "
                          << root_dir_;
}

tensorflow::Status TFRecordCheckpointer::Save(std::vector<Table*> tables,
                                              int keep_latest,
                                              std::string* path) {
  if (keep_latest <= 0) {
    return tensorflow::errors::InvalidArgument(
        "TFRecordCheckpointer must have keep_latest > 0.");
  }
  if (!group_.empty()) {
    return tensorflow::errors::InvalidArgument(
        "Setting non-empty group is not supported");
  }

  std::string dir_path =
      tensorflow::io::JoinPath(root_dir_, absl::FormatTime(absl::Now()));
  TF_RETURN_IF_ERROR(
      tensorflow::Env::Default()->RecursivelyCreateDir(dir_path));

  RecordWriterUniquePtr table_writer;
  TF_RETURN_IF_ERROR(OpenWriter(
      tensorflow::io::JoinPath(dir_path, kTablesFileName), &table_writer));

  absl::flat_hash_set<std::shared_ptr<ChunkStore::Chunk>> chunks;
  for (Table* table : tables) {
    auto checkpoint = table->Checkpoint();
    chunks.merge(checkpoint.chunks);
    TF_RETURN_IF_ERROR(
        table_writer->WriteRecord(checkpoint.checkpoint.SerializeAsString()));
  }

  TF_RETURN_IF_ERROR(table_writer->Close());
  table_writer = nullptr;

  RecordWriterUniquePtr chunk_writer;
  TF_RETURN_IF_ERROR(OpenWriter(
      tensorflow::io::JoinPath(dir_path, kChunksFileName), &chunk_writer));

  for (const auto& chunk : chunks) {
    TF_RETURN_IF_ERROR(
        chunk_writer->WriteRecord(chunk->data().SerializeAsString()));
  }
  TF_RETURN_IF_ERROR(chunk_writer->Close());
  chunk_writer = nullptr;

  // Both chunks and table checkpoint has now been written so we can proceed to
  // add the DONE-file.
  TF_RETURN_IF_ERROR(WriteDone(dir_path));

  // Delete the older checkpoints.
  std::vector<std::string> filenames;
  TF_RETURN_IF_ERROR(tensorflow::Env::Default()->GetMatchingPaths(
      tensorflow::io::JoinPath(root_dir_, "*"), &filenames));
  std::sort(filenames.begin(), filenames.end());
  int history_counter = 0;
  for (auto it = filenames.rbegin(); it != filenames.rend(); it++) {
    if (++history_counter > keep_latest) {
      tensorflow::int64 undeleted_files;
      tensorflow::int64 undeleted_dirs;
      TF_RETURN_IF_ERROR(tensorflow::Env::Default()->DeleteRecursively(
          *it, &undeleted_files, &undeleted_dirs));
    }
  }

  *path = std::move(dir_path);
  return tensorflow::Status::OK();
}

tensorflow::Status TFRecordCheckpointer::Load(
    absl::string_view relative_path, ChunkStore* chunk_store,
    std::vector<std::shared_ptr<Table>>* tables) {
  const std::string dir_path =
      tensorflow::io::JoinPath(root_dir_, relative_path);
  REVERB_LOG(REVERB_INFO) << "Loading checkpoint from " << dir_path;
  if (!HasDone(dir_path)) {
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("Load called with invalid checkpoint path: ", dir_path));
  }
  // Insert data first to ensure that all data referenced by the tables
  // exists. Keep the map of chunks around so that none of the chunks are
  // cleaned up before all the tables have been loaded.
  internal::flat_hash_map<ChunkStore::Key, std::shared_ptr<ChunkStore::Chunk>>
      chunk_by_key;
  {
    RecordReaderUniquePtr chunk_reader;
    TF_RETURN_IF_ERROR(OpenReader(
        tensorflow::io::JoinPath(dir_path, kChunksFileName), &chunk_reader));

    ChunkData chunk_data;
    tensorflow::Status chunk_status;
    tensorflow::uint64 chunk_offset = 0;
    tensorflow::tstring chunk_record;
    do {
      chunk_status = chunk_reader->ReadRecord(&chunk_offset, &chunk_record);
      if (!chunk_status.ok()) break;
      if (!chunk_data.ParseFromArray(chunk_record.data(),
                                     chunk_record.size())) {
        return tensorflow::errors::DataLoss(
            "Could not parse TFRecord as ChunkData: '", chunk_record, "'");
      }
      if (chunk_data.deprecated_data_size()) {
        if (!chunk_data.data().tensors().empty()) {
          return tensorflow::errors::Internal(
              "Checkpoint ChunkData at offset: ", chunk_offset,
              " has both data and deprecated_data.");
        }
        chunk_data.mutable_data()->mutable_tensors()->Swap(
            chunk_data.mutable_deprecated_data());
      }
      chunk_by_key[chunk_data.chunk_key()] = chunk_store->Insert(chunk_data);
    } while (chunk_status.ok());
    if (!tensorflow::errors::IsOutOfRange(chunk_status)) {
      return chunk_status;
    }
  }

  RecordReaderUniquePtr table_reader;
  TF_RETURN_IF_ERROR(OpenReader(
      tensorflow::io::JoinPath(dir_path, kTablesFileName), &table_reader));

  PriorityTableCheckpoint checkpoint;
  tensorflow::Status table_status;
  tensorflow::uint64 table_offset = 0;
  tensorflow::tstring table_record;
  do {
    table_status = table_reader->ReadRecord(&table_offset, &table_record);
    if (!table_status.ok()) break;
    if (!checkpoint.ParseFromArray(table_record.data(), table_record.size())) {
      return tensorflow::errors::DataLoss(
          "Could not parse TFRecord as Checkpoint: '", table_record, "'");
    }

    int index = find_table_index(tables, checkpoint.table_name());
    if (index == -1) {
      std::vector<std::string> table_names;
      for (const auto& table : *tables) {
        table_names.push_back(absl::StrCat("'", table->name(), "'"));
      }
      return tensorflow::errors::InvalidArgument(
          "Trying to load table ", checkpoint.table_name(),
          " but table was not found in provided list of tables. Available "
          "tables: [",
          absl::StrJoin(table_names, ", "), "]");
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

      for (const auto& key : checkpoint_item.chunk_keys()) {
        REVERB_CHECK(chunk_by_key.contains(key));
        insert_item.chunks.push_back(chunk_by_key[key]);
      }

      // The original table has already been destroyed so if this fails then
      // there is way to recover.
      TF_CHECK_OK(table->InsertCheckpointItem(std::move(insert_item)));
    }

    tables->at(index).swap(table);
  } while (table_status.ok());

  if (!tensorflow::errors::IsOutOfRange(table_status)) {
    return table_status;
  }
  return tensorflow::Status::OK();
}

tensorflow::Status TFRecordCheckpointer::LoadLatest(
    ChunkStore* chunk_store, std::vector<std::shared_ptr<Table>>* tables) {
  REVERB_LOG(REVERB_INFO) << "Loading latest checkpoint from " << root_dir_;
  std::vector<std::string> filenames;
  TF_RETURN_IF_ERROR(tensorflow::Env::Default()->GetMatchingPaths(
      tensorflow::io::JoinPath(root_dir_, "*"), &filenames));
  std::sort(filenames.begin(), filenames.end());
  for (auto it = filenames.rbegin(); it != filenames.rend(); it++) {
    if (HasDone(*it)) {
      return Load(tensorflow::io::Basename(*it), chunk_store, tables);
    }
  }
  return tensorflow::errors::NotFound(
      absl::StrCat("No checkpoint found in ", root_dir_));
}

std::string TFRecordCheckpointer::DebugString() const {
  return absl::StrCat("TFRecordCheckpointer(root_dir=", root_dir_,
                      ", group=", group_, ")");
}

}  // namespace reverb
}  // namespace deepmind
