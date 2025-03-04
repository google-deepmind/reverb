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
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/checkpointing_utils.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/tf_util.h"
#include "reverb/cc/support/trajectory_util.h"
#include "reverb/cc/table.h"
#include "reverb/cc/table_extensions/interface.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

namespace deepmind {
namespace reverb {
namespace {

constexpr char kTablesFileName[] = "tables.tfrecord";
constexpr char kItemsFileName[] = "items.tfrecord";
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
  *writer = RecordWriterUniquePtr(
      new tensorflow::io::RecordWriter(
          file_ptr,
          tensorflow::io::RecordWriterOptions::CreateRecordWriterOptions(
              tensorflow::io::compression::kZlib)),
      [file_ptr](tensorflow::io::RecordWriter* w) {
        delete w;
        delete file_ptr;
      });
  return absl::OkStatus();
}

absl::Status OpenReader(const std::string& path, RecordReaderUniquePtr* reader,
                        const std::string& compression_type) {
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  REVERB_RETURN_IF_ERROR(FromTensorflowStatus(
      tensorflow::Env::Default()->NewRandomAccessFile(path, &file)));
  auto* file_ptr = file.release();
  *reader = RecordReaderUniquePtr(
      new tensorflow::io::RecordReader(
          file_ptr,
          tensorflow::io::RecordReaderOptions::CreateRecordReaderOptions(
              compression_type)),
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

inline bool HasItems(const std::string& path) {
  return tensorflow::Env::Default()
      ->FileExists(tensorflow::io::JoinPath(path, kItemsFileName))
      .ok();
}

absl::StatusOr<size_t> GetTableIndex(
    const std::vector<std::shared_ptr<Table>>& tables,
    const std::string& name) {
  for (size_t i = 0; i < tables.size(); i++) {
    if (tables[i]->name() == name) {
      return i;
    }
  }
  std::vector<std::string> table_names(tables.size());
  for (size_t i = 0; i < tables.size(); i++) {
    table_names[i] = absl::StrCat("'", tables[i]->name(), "'");
  }

  return absl::InvalidArgumentError(absl::StrCat(
      "Trying to load table '", name,
      "' but table was not found in provided list of tables. Available "
      "tables: [",
      absl::StrJoin(table_names, ", "), "]"));
}

absl::Status CheckTrajectoryFormat(const PrioritizedItem& item) {
  if (item.has_deprecated_sequence_range() && item.has_flat_trajectory()) {
    return absl::InternalError(absl::StrCat(
        "Item ", item.key(), " has both deprecated and new trajectory format: ",
        item.DebugString(), "."));
  }
  return absl::OkStatus();
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

  internal::flat_hash_set<std::shared_ptr<ChunkStore::Chunk>> chunks;
  std::vector<PrioritizedItem> items;
  {
    RecordWriterUniquePtr table_writer;
    REVERB_RETURN_IF_ERROR(OpenWriter(
        tensorflow::io::JoinPath(dir_path, kTablesFileName), &table_writer));

    for (Table* table : tables) {
      auto checkpoint = table->Checkpoint();
      chunks.merge(checkpoint.chunks);
      items.insert(items.end(),
                   std::make_move_iterator(checkpoint.items.begin()),
                   std::make_move_iterator(checkpoint.items.end()));
      std::string serialized;
      if (!checkpoint.checkpoint.AppendToString(&serialized)) {
        return absl::DataLossError(absl::StrCat(
            "Unable to serialize checkpoint object.  Table "
            "name: '",
            checkpoint.checkpoint.table_name(),
            "' and proto size: ", checkpoint.checkpoint.ByteSizeLong(),
            " bytes. Perhaps the proto is >2GB?  Please check your logs."));
      }
      REVERB_RETURN_IF_ERROR(
          FromTensorflowStatus(table_writer->WriteRecord(serialized)));
    }

    REVERB_RETURN_IF_ERROR(FromTensorflowStatus(table_writer->Close()));
  }

  {
    RecordWriterUniquePtr item_writer;
    REVERB_RETURN_IF_ERROR(OpenWriter(
        tensorflow::io::JoinPath(dir_path, kItemsFileName), &item_writer));

    for (const auto& item : items) {
      std::string serialized;
      if (!item.AppendToString(&serialized)) {
        return absl::DataLossError(
            absl::StrCat("Unable to serialize item.  Item key: '", item.key(),
                         "' and proto size: ", item.ByteSizeLong(),
                         " bytes.  Please check your logs."));
      }
      REVERB_RETURN_IF_ERROR(
          FromTensorflowStatus(item_writer->WriteRecord(serialized)));
    }
    REVERB_RETURN_IF_ERROR(FromTensorflowStatus(item_writer->Close()));
  }

  {
    RecordWriterUniquePtr chunk_writer;
    REVERB_RETURN_IF_ERROR(OpenWriter(
        tensorflow::io::JoinPath(dir_path, kChunksFileName), &chunk_writer));

    for (const auto& chunk : chunks) {
      std::string serialized;
      if (!chunk->data().AppendToString(&serialized)) {
        return absl::DataLossError(absl::StrCat(
            "Unable to serialize chunk.  Chunk key: '", chunk->key(),
            "' and proto size: ", chunk->data().ByteSizeLong(),
            " bytes.  Perhaps the proto is >2GB?  Please also check your "
            "logs."));
      }
      REVERB_RETURN_IF_ERROR(
          FromTensorflowStatus(chunk_writer->WriteRecord(serialized)));
    }
    REVERB_RETURN_IF_ERROR(FromTensorflowStatus(chunk_writer->Close()));
  }

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
      int64_t undeleted_files;
      int64_t undeleted_dirs;
      REVERB_RETURN_IF_ERROR(
          FromTensorflowStatus(tensorflow::Env::Default()->DeleteRecursively(
              *it, &undeleted_files, &undeleted_dirs)));
    }
  }

  *path = std::move(dir_path);
  return absl::OkStatus();
}

namespace {

absl::Status LoadWithCompression(absl::string_view path,
                                 ChunkStore* chunk_store,
                                 std::vector<std::shared_ptr<Table>>* tables,
                                 const std::string& compression_type) {
  REVERB_LOG(REVERB_INFO) << "Loading checkpoint from " << std::string(path);
  if (!HasDone(std::string(path))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Load called with invalid checkpoint path: ", std::string(path)));
  }

  // Before spending (a potentially very long) time reading the data referenced
  // by the table items, let's load the (relatively small) table protos and
  // check that the checkpoint matches the tables of the server.
  REVERB_LOG(REVERB_INFO)
      << "Loading and verifying metadata of the checkpointed tables.";

  // Maps table_name to checkpoint.
  internal::flat_hash_map<std::string, PriorityTableCheckpoint>
      table_checkpoints;
  internal::flat_hash_map<std::string, std::vector<PrioritizedItem>>
      table_to_items;

  std::string deprecated_items;
  {
    RecordReaderUniquePtr table_reader;
    REVERB_RETURN_IF_ERROR(
        OpenReader(tensorflow::io::JoinPath(std::string(path), kTablesFileName),
                   &table_reader, compression_type));

    absl::Status table_status;
    uint64_t table_offset = 0;
    tensorflow::tstring table_record;
    do {
      table_status = FromTensorflowStatus(
          table_reader->ReadRecord(&table_offset, &table_record));
      if (!table_status.ok()) break;

      PriorityTableCheckpoint checkpoint;
      if (!checkpoint.ParseFromArray(table_record.data(),
                                     table_record.size())) {
        return absl::DataLossError(
            absl::StrCat("Could not parse TFRecord as Checkpoint: '",
                         absl::string_view(table_record), "'"));
      }

      REVERB_RETURN_IF_ERROR(
          GetTableIndex(*tables, checkpoint.table_name()).status());

      if (!checkpoint.deprecated_items().empty()) {
        auto& items = table_to_items[checkpoint.table_name()];
        items.reserve(checkpoint.deprecated_items().size());
        deprecated_items = absl::StrCat("'", checkpoint.table_name(), "'");
        for (auto& item : *checkpoint.mutable_deprecated_items()) {
          REVERB_RETURN_IF_ERROR(CheckTrajectoryFormat(item));
          items.push_back(std::move(item));
        }
        checkpoint.mutable_deprecated_items()->Clear();
      }

      REVERB_LOG(REVERB_INFO)
          << "Metadata for table '" << checkpoint.table_name()
          << "' was successfully loaded and verified.";
      std::string table_name = checkpoint.table_name();
      table_checkpoints[table_name] = std::move(checkpoint);
      checkpoint.Clear();
    } while (table_status.ok());

    if (!absl::IsOutOfRange(table_status)) {
      return table_status;
    }
  }

  bool non_deprecated_items = HasItems(std::string(path));

  if (non_deprecated_items) {
    if (!deprecated_items.empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Checkpoint loader found mix of deprecated_items field for table ",
          deprecated_items, " and items file '",
          tensorflow::io::JoinPath(std::string(path), kItemsFileName), "'"));
    }

    RecordReaderUniquePtr item_reader;
    REVERB_RETURN_IF_ERROR(
        OpenReader(tensorflow::io::JoinPath(std::string(path), kItemsFileName),
                   &item_reader, compression_type));

    PrioritizedItem item;
    absl::Status item_status;
    uint64_t item_offset = 0;
    tensorflow::tstring item_record;
    std::vector<PrioritizedItem>* items = nullptr;
    do {
      item_status = FromTensorflowStatus(
          item_reader->ReadRecord(&item_offset, &item_record));
      if (!item_status.ok()) break;
      if (!item.ParseFromArray(item_record.data(), item_record.size())) {
        return absl::DataLossError(
            absl::StrCat("Could not parse TFRecord as PrioritizedItem: '",
                         absl::string_view(item_record), "'"));
      }

      REVERB_RETURN_IF_ERROR(CheckTrajectoryFormat(item));

      if (items == nullptr || items->empty() ||
          items->at(0).table() != item.table()) {
        if (!table_checkpoints.contains(item.table())) {
          return absl::DataLossError(absl::StrCat(
              "Unable to find table '", item.table(), "' for item '",
              item.key(), "' in the set of tables loaded from metadata."));
        }
        items = &(table_to_items[item.table()]);
      }

      items->push_back(std::move(item));
    } while (item_status.ok());

    if (!absl::IsOutOfRange(item_status)) {
      return item_status;
    }
  }

  REVERB_LOG(REVERB_INFO)
      << "Successfully loaded and verified metadata for all ("
      << table_checkpoints.size()
      << ") tables. We'll now proceed to read the data referenced by the items "
         "in the table.";

  // Insert data first to ensure that all data referenced by the tables
  // exists. Keep the map of chunks around so that none of the chunks are
  // cleaned up before all the tables have been loaded.
  internal::flat_hash_map<ChunkStore::Key, std::shared_ptr<ChunkStore::Chunk>>
      chunk_by_key;
  {
    RecordReaderUniquePtr chunk_reader;
    REVERB_RETURN_IF_ERROR(
        OpenReader(tensorflow::io::JoinPath(std::string(path), kChunksFileName),
                   &chunk_reader, compression_type));

    ChunkData chunk_data;
    absl::Status chunk_status;
    uint64_t chunk_offset = 0;
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
      const auto key = chunk_data.chunk_key();
      chunk_by_key[key] = chunk_store->Insert(std::move(chunk_data));

      REVERB_LOG_EVERY_N(REVERB_INFO, 100)
          << "Still reading compressed trajectory data. " << chunk_by_key.size()
          << " records have been read so far.";
    } while (chunk_status.ok());

    if (!absl::IsOutOfRange(chunk_status)) {
      return chunk_status;
    }
  }

  REVERB_LOG(REVERB_INFO)
      << "Completed reading compressed trajectory data. We'll now start "
         "assembling the checkpointed tables.";

  std::vector<std::shared_ptr<TableExtension>> all_table_extensions;

  for (auto& checkpoint_ref : table_checkpoints) {
    auto& checkpoint = checkpoint_ref.second;
    REVERB_ASSIGN_OR_RETURN(size_t table_idx,
                            GetTableIndex(*tables, checkpoint.table_name()));
    std::shared_ptr<Table>& server_table = tables->at(table_idx);

    auto sampler = MakeSelector(checkpoint.sampler());
    auto remover = MakeSelector(checkpoint.remover());
    auto rate_limiter =
        std::make_shared<RateLimiter>(checkpoint.rate_limiter());
    auto signature =
        checkpoint.has_signature()
            ? absl::make_optional(std::move(checkpoint.signature()))
            : absl::nullopt;

    std::vector<std::shared_ptr<TableExtension>> extensions =
        server_table->GetExtensions();
    // Append the extensions from this table to the list of all extensions.
    all_table_extensions.insert(all_table_extensions.end(),
                                std::make_move_iterator(extensions.begin()),
                                std::make_move_iterator(extensions.end()));

    server_table->InitializeFromCheckpoint(
        std::move(sampler), std::move(remover), checkpoint.max_size(),
        checkpoint.max_times_sampled(), std::move(rate_limiter),
        std::move(signature), checkpoint.num_deleted_episodes(),
        checkpoint.num_unique_samples());
  }

  // Notify the extensions about the updated list of tables so they can update
  // their target table pointers (if any). It is important that we do this
  // before the items are loaded to allow the extensions to handle `OnInsert`
  // correctly.
  for (auto& extension : all_table_extensions) {
    extension->OnCheckpointLoaded(*tables);
  }

  for (auto& table : *tables) {
    for (auto& checkpoint_item : table_to_items[table->name()]) {
      if (checkpoint_item.has_deprecated_sequence_range()) {
        std::vector<std::shared_ptr<ChunkStore::Chunk>> trajectory_chunks;
        REVERB_RETURN_IF_ERROR(chunk_store->Get(
            checkpoint_item.deprecated_chunk_keys(), &trajectory_chunks));

        *checkpoint_item.mutable_flat_trajectory() =
            internal::FlatTimestepTrajectory(
                trajectory_chunks,
                checkpoint_item.deprecated_sequence_range().offset(),
                checkpoint_item.deprecated_sequence_range().length());

        checkpoint_item.clear_deprecated_sequence_range();
        checkpoint_item.clear_deprecated_chunk_keys();
      }

      std::vector<std::shared_ptr<ChunkStore::Chunk>> chunks;
      REVERB_RETURN_IF_ERROR(chunk_store->Get(
          internal::GetChunkKeys(checkpoint_item.flat_trajectory()), &chunks));

      // The original table has already been destroyed so if this fails then
      // there is way to recover.
      REVERB_RETURN_IF_ERROR(table->InsertCheckpointItem(
          Table::Item(std::move(checkpoint_item), std::move(chunks))));
    }

    REVERB_LOG(REVERB_INFO)
        << "Table " << table->name() << " and " << table->size()
        << " items have been successfully loaded from checkpoint at path "
        << path << ".";
  }

  REVERB_LOG(REVERB_INFO) << "Successfully loaded " << table_checkpoints.size()
                          << " tables from " << path;

  return absl::OkStatus();
}
}  // end namespace

absl::Status TFRecordCheckpointer::Load(
    absl::string_view path, ChunkStore* chunk_store,
    std::vector<std::shared_ptr<Table>>* tables) {
  auto status = LoadWithCompression(
        path, chunk_store, tables,
        /*compression_type=*/tensorflow::io::compression::kZlib);
  if (absl::IsDataLoss(status)) {
    // This may be an old checkpoint, written without compression.  Try again.
    status = LoadWithCompression(
        path, chunk_store, tables,
        /*compression_type=*/tensorflow::io::compression::kNone);
  }
  return status;
}

absl::Status TFRecordCheckpointer::LoadLatest(
    std::vector<std::shared_ptr<Table>>* tables) {
  ChunkStore chunk_store;
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
          &chunk_store, tables);
    }
  }
  return absl::NotFoundError(
      absl::StrCat("No checkpoint found in ", root_dir_));
}

absl::Status TFRecordCheckpointer::LoadFallbackCheckpoint(
    std::vector<std::shared_ptr<Table>>* tables) {
  ChunkStore chunk_store;
  if (!fallback_checkpoint_path_.has_value()) {
    return absl::NotFoundError("No fallback checkpoint path provided.");
  }
  if (HasDone(fallback_checkpoint_path_.value())) {
    return Load(fallback_checkpoint_path_.value(), &chunk_store, tables);
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
