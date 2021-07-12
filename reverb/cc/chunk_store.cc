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

#include <memory>
#include <utility>
#include <vector>

#include <cstdint>
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/thread.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/queue.h"
#include "reverb/cc/tensor_compression.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {

ChunkStore::Chunk::Chunk(ChunkData data) : data_(std::move(data)) {}

uint64_t ChunkStore::Chunk::key() const { return data_.chunk_key(); }

const ChunkData& ChunkStore::Chunk::data() const { return data_; }

size_t ChunkStore::Chunk::DataByteSizeLong() const {
  absl::call_once(data_byte_size_once_,
                  [this]() { data_byte_size_ = data_.ByteSizeLong(); });
  return data_byte_size_;
}

uint64_t ChunkStore::Chunk::episode_id() const {
  return data_.sequence_range().episode_id();
}

int32_t ChunkStore::Chunk::num_rows() const {
  return data_.sequence_range().end() - data_.sequence_range().start() + 1;
}

int ChunkStore::Chunk::num_columns() const {
  // Try to get number of columns without parsing lazy tensors field.
  if (data_.data_tensors_len() != 0) {
    return data_.data_tensors_len();
  }
  return data_.data().tensors_size();
}

ChunkStore::ChunkStore(int cleanup_batch_size)
    : delete_keys_(std::make_shared<internal::Queue<Key>>(10000000)),
      cleaner_(internal::StartThread(
          "ChunkStore-Cleaner", [this, cleanup_batch_size] {
            while (CleanupInternal(cleanup_batch_size)) {
            }
          })) {}

ChunkStore::~ChunkStore() {
  // Closing the queue makes all calls to `CleanupInternal` to return false
  // which will break the loop in `cleaner_` making it joinable.
  delete_keys_->Close();
  cleaner_ = nullptr;  // Joins thread.
}

std::shared_ptr<ChunkStore::Chunk> ChunkStore::Insert(ChunkData item) {
  absl::WriterMutexLock lock(&mu_);
  std::weak_ptr<Chunk>& wp = data_[item.chunk_key()];
  std::shared_ptr<Chunk> sp = wp.lock();
  if (sp == nullptr) {
    wp = (sp = std::shared_ptr<Chunk>(new Chunk(std::move(item)),
                                      [q = delete_keys_](Chunk* chunk) {
                                        q->Push(chunk->key());
                                        delete chunk;
                                      }));
  }
  return sp;
}

tensorflow::Status ChunkStore::Get(
    absl::Span<const ChunkStore::Key> keys,
    std::vector<std::shared_ptr<Chunk>>* chunks) {
  absl::ReaderMutexLock lock(&mu_);
  chunks->clear();
  chunks->reserve(keys.size());
  for (int i = 0; i < keys.size(); i++) {
    chunks->push_back(GetItem(keys[i]));
    if (!chunks->at(i)) {
      return tensorflow::errors::NotFound(
          absl::StrCat("Chunk ", keys[i], " cannot be found."));
    }
  }
  return tensorflow::Status::OK();
}

std::shared_ptr<ChunkStore::Chunk> ChunkStore::GetItem(Key key) {
  auto it = data_.find(key);
  return it == data_.end() ? nullptr : it->second.lock();
}

bool ChunkStore::CleanupInternal(int num_chunks) {
  std::vector<Key> popped_keys;
  popped_keys.reserve(num_chunks);
  if (!delete_keys_->PopBatch(num_chunks, &popped_keys).ok()) {
    return false;
  }

  absl::WriterMutexLock data_lock(&mu_);
  for (const Key& key : popped_keys) {
    data_.erase(key);
  }

  return true;
}

}  // namespace reverb
}  // namespace deepmind
