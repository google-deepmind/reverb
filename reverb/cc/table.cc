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

#include "reverb/cc/table.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/timestamp.pb.h"
#include <cstdint>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/interface.h"
#include "reverb/cc/table_extensions/interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/status.h"

namespace deepmind {
namespace reverb {
namespace {

using Extensions = std::vector<std::shared_ptr<TableExtensionInterface>>;

inline bool IsAdjacent(const SequenceRange& a, const SequenceRange& b) {
  return a.episode_id() == b.episode_id() && a.end() + 1 == b.start();
}

inline bool IsInsertedBefore(const PrioritizedItem& a,
                             const PrioritizedItem& b) {
  return a.inserted_at().seconds() < b.inserted_at().seconds() ||
         (a.inserted_at().seconds() == b.inserted_at().seconds() &&
          a.inserted_at().nanos() < b.inserted_at().nanos());
}

inline void EncodeAsTimestampProto(absl::Time t,
                                   google::protobuf::Timestamp* proto) {
  const int64_t s = absl::ToUnixSeconds(t);
  proto->set_seconds(s);
  proto->set_nanos((t - absl::FromUnixSeconds(s)) / absl::Nanoseconds(1));
}

}  // namespace

Table::Table(std::string name, std::shared_ptr<ItemSelectorInterface> sampler,
             std::shared_ptr<ItemSelectorInterface> remover, int64_t max_size,
             int32_t max_times_sampled, std::shared_ptr<RateLimiter> rate_limiter,
             Extensions extensions,
             absl::optional<tensorflow::StructuredValue> signature)
    : sampler_(std::move(sampler)),
      remover_(std::move(remover)),
      max_size_(max_size),
      max_times_sampled_(max_times_sampled),
      name_(std::move(name)),
      rate_limiter_(std::move(rate_limiter)),
      extensions_(std::move(extensions)),
      signature_(std::move(signature)) {
  TF_CHECK_OK(rate_limiter_->RegisterTable(this));
  for (auto& extension : extensions_) {
    TF_CHECK_OK(extension->RegisterTable(&mu_, this));
  }
}

Table::~Table() {
  rate_limiter_->UnregisterTable(&mu_, this);
  for (auto& extension : extensions_) {
    extension->UnregisterTable(&mu_, this);
  }
}

std::vector<Table::Item> Table::Copy(size_t count) const {
  std::vector<Item> items;
  absl::ReaderMutexLock lock(&mu_);
  items.reserve(count == 0 ? data_.size() : count);
  for (auto it = data_.cbegin();
       it != data_.cend() && (count == 0 || items.size() < count); it++) {
    items.push_back(it->second);
  }
  return items;
}

tensorflow::Status Table::InsertOrAssign(Item item) {
  auto key = item.item.key();
  auto priority = item.item.priority();

  // If an item is deleted as part of the insert then we keep the data alive
  // until the lock has been released.
  Item deleted_item;
  {
    absl::WriterMutexLock lock(&mu_);

    /// If item already exists in table then update its priority.
    if (data_.contains(key)) {
      return UpdateItem(key, priority);
    }

    // Wait for the insert to be staged. While waiting the lock is released but
    // once it returns the lock is aquired again. While waiting for the right to
    // insert the operation might have transformed into an update.
    TF_RETURN_IF_ERROR(rate_limiter_->AwaitCanInsert(&mu_));

    if (data_.contains(key)) {
      // If the insert was transformed into an update while waiting we need to
      // notify the limiter so it let another insert call to proceed.
      rate_limiter_->MaybeSignalCondVars(&mu_);
      return UpdateItem(key, priority);
    }

    // Set the insertion timestamp after the lock has been acquired as this
    // represents the order it was inserted into the sampler and remover.
    EncodeAsTimestampProto(absl::Now(), item.item.mutable_inserted_at());
    data_[key] = std::move(item);

    TF_RETURN_IF_ERROR(sampler_->Insert(key, priority));
    TF_RETURN_IF_ERROR(remover_->Insert(key, priority));

    auto it = data_.find(key);
    for (auto& extension : extensions_) {
      extension->OnInsert(&mu_, it->second);
    }

    // Increment references to the episode/s the item is referencing.
    // We increment before a possible call to DeleteItem since the sampler can
    // return this key.
    for (const auto& chunk : it->second.chunks) {
      ++episode_refs_[chunk->data().sequence_range().episode_id()];
    }

    // Remove an item if we exceeded `max_size_`.
    if (data_.size() > max_size_) {
      TF_RETURN_IF_ERROR(DeleteItem(remover_->Sample().key, &deleted_item));
    }

    // Now that the new item has been inserted and an older item has
    // (potentially) been removed the insert can be finalized.
    rate_limiter_->Insert(&mu_);
  }

  return tensorflow::Status::OK();
}

tensorflow::Status Table::MutateItems(absl::Span<const KeyWithPriority> updates,
                                      absl::Span<const Key> deletes) {
  std::vector<Item> deleted_items(deletes.size());
  {
    absl::WriterMutexLock lock(&mu_);
    for (int i = 0; i < deletes.size(); i++) {
      TF_RETURN_IF_ERROR(DeleteItem(deletes[i], &deleted_items[i]));
    }
    for (const auto& item : updates) {
      TF_RETURN_IF_ERROR(UpdateItem(item.key(), item.priority()));
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status Table::Sample(SampledItem* sampled_item,
                                 absl::Duration timeout) {
  // Keep references to the (potentially) deleted item alive until the lock has
  // been released.
  Item deleted_item;
  {
    absl::WriterMutexLock lock(&mu_);
    TF_RETURN_IF_ERROR(rate_limiter_->AwaitAndFinalizeSample(&mu_, timeout));

    auto sample = sampler_->Sample();
    Item& item = data_[sample.key];

    // Increment the sample count.
    item.item.set_times_sampled(item.item.times_sampled() + 1);

    // Copy Details of the sampled item.
    sampled_item->item = item.item;
    sampled_item->chunks = item.chunks;
    sampled_item->probability = sample.probability;
    sampled_item->table_size = data_.size();

    // Notify extensions which item was sampled.
    for (auto& extension : extensions_) {
      extension->OnSample(&mu_, item);
    }

    // If there is an upper bound of the number of times an item can be sampled
    // and it is now reached then delete the item before the lock is released.
    if (item.item.times_sampled() == max_times_sampled_) {
      TF_RETURN_IF_ERROR(DeleteItem(item.item.key(), &deleted_item));
    }
  }

  return tensorflow::Status::OK();
}

int64_t Table::size() const {
  absl::ReaderMutexLock lock(&mu_);
  return data_.size();
}

const std::string& Table::name() const { return name_; }

TableInfo Table::info() const {
  TableInfo info;

  info.set_name(name_);
  info.set_max_size(max_size_);
  info.set_max_times_sampled(max_times_sampled_);

  if (signature_) {
    *info.mutable_signature() = *signature_;
  }

  absl::ReaderMutexLock lock(&mu_);
  *info.mutable_rate_limiter_info() = rate_limiter_->Info(&mu_);
  *info.mutable_sampler_options() = sampler_->options();
  *info.mutable_remover_options() = remover_->options();
  info.set_current_size(data_.size());
  info.set_num_episodes(episode_refs_.size());

  return info;
}

void Table::Close() {
  absl::WriterMutexLock lock(&mu_);
  rate_limiter_->Cancel(&mu_);
}

tensorflow::Status Table::DeleteItem(Table::Key key, Item* deleted_item) {
  auto it = data_.find(key);
  if (it == data_.end()) return tensorflow::Status::OK();

  for (auto& extension : extensions_) {
    extension->OnDelete(&mu_, it->second);
  }

  // Decrement counts to the episodes the item is referencing.
  for (const auto& chunk : it->second.chunks) {
    auto ep_it =
        episode_refs_.find(chunk->data().sequence_range().episode_id());
    REVERB_CHECK(ep_it != episode_refs_.end());
    if (--(ep_it->second) == 0) {
      episode_refs_.erase(ep_it);
    }
  }

  *deleted_item = std::move(it->second);
  data_.erase(it);
  rate_limiter_->Delete(&mu_);
  TF_RETURN_IF_ERROR(sampler_->Delete(key));
  TF_RETURN_IF_ERROR(remover_->Delete(key));
  return tensorflow::Status::OK();
}

tensorflow::Status Table::UpdateItem(
    Key key, double priority,
    std::initializer_list<TableExtensionInterface*> exclude) {
  auto it = data_.find(key);
  if (it == data_.end()) {
    return tensorflow::Status::OK();
  }
  it->second.item.set_priority(priority);
  TF_RETURN_IF_ERROR(sampler_->Update(key, priority));
  TF_RETURN_IF_ERROR(remover_->Update(key, priority));

  for (auto& extension : extensions_) {
    if (std::none_of(
            exclude.begin(), exclude.end(),
            [ext_ptr = extension.get()](auto e) { return e == ext_ptr; })) {
      extension->OnUpdate(&mu_, it->second);
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status Table::Reset() {
  absl::WriterMutexLock lock(&mu_);

  for (auto& extension : extensions_) {
    extension->OnReset(&mu_);
  }

  sampler_->Clear();
  remover_->Clear();

  data_.clear();

  rate_limiter_->Reset(&mu_);

  return tensorflow::Status::OK();
}

Table::CheckpointAndChunks Table::Checkpoint() {
  PriorityTableCheckpoint checkpoint;
  checkpoint.set_table_name(name());
  checkpoint.set_max_size(max_size_);
  checkpoint.set_max_times_sampled(max_times_sampled_);

  absl::ReaderMutexLock lock(&mu_);

  *checkpoint.mutable_sampler() = sampler_->options();
  *checkpoint.mutable_remover() = remover_->options();

  // Note that is is important that the rate limiter checkpoint is
  // finalized before the items are added
  *checkpoint.mutable_rate_limiter() = rate_limiter_->CheckpointReader(&mu_);

  absl::flat_hash_set<std::shared_ptr<ChunkStore::Chunk>> chunks;
  for (const auto& entry : data_) {
    *checkpoint.add_items() = entry.second.item;
    chunks.insert(entry.second.chunks.begin(), entry.second.chunks.end());
  }

  // Sort the items in ascending order based on their insertion time. This makes
  // it possible to reconstruct ordered structures (Fifo) when the checkpoint is
  // loaded.
  std::sort(checkpoint.mutable_items()->begin(),
            checkpoint.mutable_items()->end(), IsInsertedBefore);

  return {std::move(checkpoint), std::move(chunks)};
}

tensorflow::Status Table::InsertCheckpointItem(Table::Item item) {
  absl::WriterMutexLock lock(&mu_);
  REVERB_CHECK_LE(data_.size() + 1, max_size_)
      << "InsertCheckpointItem called on already full Table";
  REVERB_CHECK(!data_.contains(item.item.key()))
      << "InsertCheckpointItem called for item with already present key: "
      << item.item.key();

  TF_RETURN_IF_ERROR(sampler_->Insert(item.item.key(), item.item.priority()));
  TF_RETURN_IF_ERROR(remover_->Insert(item.item.key(), item.item.priority()));

  const auto key = item.item.key();
  auto it = data_.emplace(key, std::move(item)).first;
  for (auto& extension : extensions_) {
    extension->OnInsert(&mu_, it->second);
  }

  for (const auto& chunk : it->second.chunks) {
    ++episode_refs_[chunk->data().sequence_range().episode_id()];
  }

  return tensorflow::Status::OK();
}

bool Table::Get(Table::Key key, Table::Item* item) {
  absl::ReaderMutexLock lock(&mu_);
  auto it = data_.find(key);
  if (it != data_.end()) {
    *item = it->second;
    return true;
  }
  return false;
}

const absl::flat_hash_map<Table::Key, Table::Item>* Table::RawLookup() {
  mu_.AssertReaderHeld();
  return &data_;
}

void Table::UnsafeAddExtension(
    std::shared_ptr<TableExtensionInterface> extension) {
  TF_CHECK_OK(extension->RegisterTable(&mu_, this));
  absl::WriterMutexLock lock(&mu_);
  REVERB_CHECK(data_.empty());
  extensions_.push_back(std::move(extension));
}

const std::vector<std::shared_ptr<TableExtensionInterface>>& Table::extensions()
    const {
  return extensions_;
}

const absl::optional<tensorflow::StructuredValue>& Table::signature() const {
  return signature_;
}

bool Table::CanSample(int num_samples) const {
  absl::ReaderMutexLock lock(&mu_);
  return rate_limiter_->CanSample(&mu_, num_samples);
}

bool Table::CanInsert(int num_inserts) const {
  absl::ReaderMutexLock lock(&mu_);
  return rate_limiter_->CanInsert(&mu_, num_inserts);
}

RateLimiterEventHistory Table::GetRateLimiterEventHistory(
    size_t min_insert_event_id, size_t min_sample_event_id) const {
  absl::ReaderMutexLock lock(&mu_);
  return rate_limiter_->GetEventHistory(&mu_, min_insert_event_id,
                                        min_sample_event_id);
}

int64_t Table::num_episodes() const {
  absl::ReaderMutexLock lock(&mu_);
  return episode_refs_.size();
}

tensorflow::Status Table::UnsafeUpdateItem(
    Key key, double priority,
    std::initializer_list<TableExtensionInterface*> exclude) {
  mu_.AssertHeld();
  return UpdateItem(key, priority, std::move(exclude));
}

std::vector<std::shared_ptr<TableExtensionInterface>>
Table::UnsafeClearExtensions() {
  std::vector<std::shared_ptr<TableExtensionInterface>> extensions;
  {
    absl::WriterMutexLock lock(&mu_);
    REVERB_CHECK(data_.empty());
    extensions.swap(extensions_);
  }

  for (auto& extension : extensions) {
    extension->UnregisterTable(&mu_, this);
  }

  return extensions;
}

}  // namespace reverb
}  // namespace deepmind
