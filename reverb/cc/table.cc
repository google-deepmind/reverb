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
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/errors.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/interface.h"
#include "reverb/cc/support/trajectory_util.h"
#include "reverb/cc/table_extensions/interface.h"

namespace deepmind {
namespace reverb {
namespace {

using Extensions = std::vector<std::shared_ptr<TableExtension>>;

// Number of queued inserts that are allowed on the table without slowing down
// further inserts.
static constexpr int64_t kMaxEnqueuedInserts = 1000;

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

inline absl::Status CheckItemValidity(const Table::Item& item) {
  if (item.item.flat_trajectory().columns().empty() ||
      item.item.flat_trajectory().columns(0).chunk_slices().empty()) {
    return absl::InvalidArgumentError("Item trajectory must not be empty.");
  }

  auto trajectory_keys = internal::GetChunkKeys(item.item.flat_trajectory());
  if (trajectory_keys.size() != item.chunks.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("The number of chunks (", item.chunks.size(),
                     ") does not equal the number of chunks referenced "
                     "in item's trajectory (",
                     trajectory_keys.size(), ")."));
  }

  for (int i = 0; i < trajectory_keys.size(); ++i) {
    if (item.chunks[i]->key() != trajectory_keys[i]) {
      return absl::InvalidArgumentError(
          "Item chunks does not match chunks referenced in trajectory.");
    }
  }

  return absl::OkStatus();
}

}  // namespace

// Finalize sampling request with a given status.
void FinalizeSampleRequest(std::unique_ptr<Table::SampleRequest> request,
                           absl::Status status) {
  request->status = status;
  auto to_notify = request->on_batch_done.lock();
  // Callback might have been destroyed in the meantime.
  if (to_notify != nullptr) {
    (*to_notify)(std::move(request));
  }
}

// For a given set of sampling requests extract the ones exceeding provided
// deadline. Compute `next_deadline` when some request should be terminated
// in the future. Requests to be terminated are moved into `to_terminate`
// and replaced by nullptr.
void GetExpiredRequests(
    const absl::Time& deadline,
    std::vector<std::unique_ptr<Table::SampleRequest>>* requests,
    std::vector<std::unique_ptr<Table::SampleRequest>>* to_terminate,
    absl::Time* next_deadline) {
  for (auto& sample : *requests) {
    if (sample == nullptr) {
      // Request has already been removed during the previous iteration.
      continue;
    }
    if (sample->deadline <= deadline) {
      to_terminate->push_back(std::move(sample));
    } else {
      *next_deadline = std::min(*next_deadline, sample->deadline);
    }
  }
}

Table::Table(std::string name, std::shared_ptr<ItemSelector> sampler,
             std::shared_ptr<ItemSelector> remover, int64_t max_size,
             int32_t max_times_sampled, std::shared_ptr<RateLimiter> rate_limiter,
             Extensions extensions,
             absl::optional<tensorflow::StructuredValue> signature)
    : sampler_(std::move(sampler)),
      remover_(std::move(remover)),
      num_deleted_episodes_(0),
      num_unique_samples_(0),
      max_size_(max_size),
      max_times_sampled_(max_times_sampled),
      name_(std::move(name)),
      rate_limiter_(std::move(rate_limiter)),
      extensions_(std::move(extensions)),
      signature_(std::move(signature)) {
  REVERB_CHECK_OK(rate_limiter_->RegisterTable(this));
  for (auto& extension : extensions_) {
    REVERB_CHECK_OK(extension->RegisterTable(&mu_, this));
  }
  table_worker_ = internal::StartThread("TableWorker_" + name, [&] {
    // Sampling requests that exceeded deadline and should be terminated.
    std::vector<std::unique_ptr<Table::SampleRequest>> to_terminate;
    // Status to be send to the sample requests to be terminated (changes upon
    // table shutdown).
    absl::Status terminate_status = errors::RateLimiterTimeout();
    // Collection of items removed from the table (by either insert or sample
    // operations) which are waiting for cleanup. Cleanup is done outside of
    // time-critical table worker.
    std::vector<std::shared_ptr<Item>> to_delete;
    // Collection of callbacks for insert operations waiting for more space in
    // the table.
    std::vector<std::weak_ptr<std::function<void()>>> notify_inserts_ok;
    // Collection of items waiting to the added to the table.
    std::vector<std::shared_ptr<Item>> current_inserts;
    // Index of the next item in the `pending_inserts` to be processed.
    int current_insert = 0;
    // Collection of sample requests to be processed.
    std::vector<std::unique_ptr<SampleRequest>> current_sampling;
    // Index of the next request from the `sampling_requests` to be processed.
    int current_sample = 0;

    // Progress of handling insert/sample requests. Used for detecting whether
    // worker has something to do (making progress) or should go to sleep.
    int64_t progress = 0;
    int64_t last_progress = 0;
    while (true) {
      // Notify clients waiting to insert
      for (auto& notify : notify_inserts_ok) {
        auto to_notify = notify.lock();
        // Callback might have been destroyed in the meantime.
        if (to_notify != nullptr) {
          (*to_notify)();
        }
      }
      notify_inserts_ok.clear();
      {
        absl::MutexLock lock(&mu_);
        // Tracks whether while loop below makes progress.
        int64_t prev_progress = progress - 1;
        while (prev_progress < progress) {
          prev_progress = progress;
          // Try processing an insert request.
          if (current_insert < current_inserts.size() &&
              rate_limiter_->CanInsert(&mu_, 1)) {
            auto res = InsertOrAssignInternal(current_inserts[current_insert++],
                                              &to_delete);
            progress++;
            if (!res.ok()) {
              REVERB_LOG(REVERB_ERROR)
                  << "Table worker encountered an error while inserting: "
                  << res.ToString() << ". Shutting down.";
              return;
            }
          }
          // Skip sampling requests which timed out already.
          while (current_sample < current_sampling.size() &&
                 current_sampling[current_sample] == nullptr) {
            current_sample++;
          }
          // Try processing a sample request.
          if (current_sample < current_sampling.size()) {
            auto& request = current_sampling[current_sample];
            while (rate_limiter_->MaybeCommitSample(&mu_)) {
              request->samples.emplace_back();
              auto res = SampleInternal(&request->samples.back(), &to_delete);
              if (!res.ok()) {
                REVERB_LOG(REVERB_ERROR)
                    << "Table worker encountered an error while sampling: "
                    << res.ToString() << ". Shutting down.";
                return;
              }
              // Capacity of the samples collection indicates how many items
              // should be sampled.
              if (request->samples.capacity() == request->samples.size()) {
                // Finalized request is moved out of sampling_requests.
                FinalizeSampleRequest(std::move(request), absl::OkStatus());
                current_sample++;
                break;
              }
            }
          }
        }
      }
      {
        absl::MutexLock lock(&worker_mu_);
        // Items to be deleted are being deleted by the callers to spread the
        // load. We could consider doing that in a dedicated thread instead.
        if (!to_delete.empty() && deleted_items_.empty()) {
          std::swap(to_delete, deleted_items_);
        }
        if (current_insert == current_inserts.size() &&
            !pending_inserts_.empty()) {
          // Get a new batch of insert requests as previous batch is done.
          progress++;
          current_insert = 0;
          current_inserts.clear();
          std::swap(current_inserts, pending_inserts_);
          // As `pending_inserts_` is empty now, we should let waiting users
          // know it is fine to continue with the inserts.
          std::swap(notify_inserts_ok, notify_inserts_ok_);
        }
        if (current_sample == current_sampling.size() &&
            !pending_sampling_.empty()) {
          // Get a new batch of sample requests as previous batch is done.
          progress++;
          current_sample = 0;
          current_sampling.clear();
          std::swap(current_sampling, pending_sampling_);
        }
        if (!stop_worker_ && progress != last_progress) {
          // There was progress executing insert/sample requests,
          // so continue without handling timeouts.
          last_progress = progress;
          continue;
        }
        auto deadline = absl::Now();
        auto wakeup = absl::InfiniteFuture();
        if (stop_worker_) {
          deadline = absl::InfinitePast();
          terminate_status =
              absl::CancelledError("RateLimiter has been cancelled");
        }
        GetExpiredRequests(deadline, &current_sampling, &to_terminate,
                           &wakeup);
        GetExpiredRequests(deadline, &pending_sampling_, &to_terminate,
                           &wakeup);
        if (to_terminate.empty()) {
          if (stop_worker_) {
            return;
          }
          worker_sleeps_ = true;
          wakeup_worker_.WaitWithDeadline(&worker_mu_, wakeup);
          worker_sleeps_ = false;
        }
      }
      for (auto& sample : to_terminate) {
        FinalizeSampleRequest(std::move(sample), terminate_status);
      }
      to_terminate.clear();
    }
  });
}

Table::~Table() {
  Close();
  rate_limiter_->UnregisterTable(&mu_, this);
  for (auto& extension : extensions_) {
    extension->UnregisterTable(&mu_, this);
  }
}

std::vector<Table::Item> Table::Copy(size_t count) const {
  std::vector<Item> items;
  absl::MutexLock lock(&mu_);
  items.reserve(count == 0 ? data_.size() : count);
  for (auto it = data_.cbegin();
       it != data_.cend() && (count == 0 || items.size() < count); it++) {
    items.push_back(*it->second);
  }
  return items;
}

absl::Status Table::InsertOrAssign(Item item, absl::Duration timeout) {
  REVERB_RETURN_IF_ERROR(CheckItemValidity(item));

  auto key = item.item.key();
  auto priority = item.item.priority();

  // If an item is deleted as part of the insert then we keep the data alive
  // until the lock has been released.
  std::shared_ptr<Item> deleted_item;
  {
    absl::MutexLock lock(&mu_);

    /// If item already exists in table then update its priority.
    if (data_.contains(key)) {
      return UpdateItem(key, priority);
    }

    // Wait for the insert to be staged. While waiting the lock is released but
    // once it returns the lock is acquired again. While waiting for the right
    // to insert the operation might have transformed into an update.
    REVERB_RETURN_IF_ERROR(rate_limiter_->AwaitCanInsert(&mu_, timeout));

    if (data_.contains(key)) {
      // If the insert was transformed into an update while waiting we need to
      // notify the limiter so it let another insert call to proceed.
      rate_limiter_->MaybeSignalCondVars(&mu_);
      return UpdateItem(key, priority);
    }

    // Set the insertion timestamp after the lock has been acquired as this
    // represents the order it was inserted into the sampler and remover.
    EncodeAsTimestampProto(absl::Now(), item.item.mutable_inserted_at());
    data_[key] = std::make_shared<Item>(std::move(item));

    REVERB_RETURN_IF_ERROR(sampler_->Insert(key, priority));
    REVERB_RETURN_IF_ERROR(remover_->Insert(key, priority));

    auto it = data_.find(key);
    for (auto& extension : extensions_) {
      extension->OnInsert(&mu_, *it->second);
    }

    // Increment references to the episode/s the item is referencing.
    // We increment before a possible call to DeleteItem since the sampler can
    // return this key.
    for (const auto& chunk : it->second->chunks) {
      ++episode_refs_[chunk->episode_id()];
    }

    // Remove an item if we exceeded `max_size_`.
    if (data_.size() > max_size_) {
      REVERB_RETURN_IF_ERROR(DeleteItem(remover_->Sample().key, &deleted_item));
    }

    // Now that the new item has been inserted and an older item has
    // (potentially) been removed the insert can be finalized.
    rate_limiter_->Insert(&mu_);
  }

  return absl::OkStatus();
}

absl::Status Table::InsertOrAssignAsync(Item item, bool* can_insert_more,
    std::weak_ptr<std::function<void()>> insert_more_callback) {
  REVERB_RETURN_IF_ERROR(CheckItemValidity(item));

  std::shared_ptr<Item> to_delete;
  {
    absl::MutexLock lock(&worker_mu_);
    pending_inserts_.push_back(std::make_shared<Item>(std::move(item)));
    if (worker_sleeps_) {
      wakeup_worker_.Signal();
    }
    if (!deleted_items_.empty()) {
      to_delete = std::move(deleted_items_.back());
      deleted_items_.pop_back();
    }
    *can_insert_more = pending_inserts_.size() < kMaxEnqueuedInserts;
    if (!*can_insert_more) {
      // Caller is not allowed to do any more inserts immediately.
      // A callback is registered, so that when there is space for more
      // requests, client is notified.
      notify_inserts_ok_.push_back(insert_more_callback);
    }
  }
  return absl::OkStatus();
}

absl::Status Table::InsertOrAssignInternal(std::shared_ptr<Item> item,
    std::vector<std::shared_ptr<Item>>* to_delete) {
  const auto key = item->item.key();
  const auto priority = item->item.priority();
  if (data_.contains(key)) {
    REVERB_RETURN_IF_ERROR(UpdateItem(key, priority));
    to_delete->push_back(std::move(item));
    return absl::OkStatus();
  }

  // Set the insertion timestamp after the lock has been acquired as this
  // represents the order it was inserted into the sampler and remover.
  EncodeAsTimestampProto(absl::Now(), item->item.mutable_inserted_at());
  data_[key] = std::move(item);

  REVERB_RETURN_IF_ERROR(sampler_->Insert(key, priority));
  REVERB_RETURN_IF_ERROR(remover_->Insert(key, priority));

  auto it = data_.find(key);
  for (auto& extension : extensions_) {
    extension->OnInsert(&mu_, *it->second);
  }

  // Increment references to the episode/s the item is referencing.
  // We increment before a possible call to DeleteItem since the sampler can
  // return this key.
  for (const auto& chunk : it->second->chunks) {
    ++episode_refs_[chunk->episode_id()];
  }

  // Remove an item if we exceeded `max_size_`.
  if (data_.size() > max_size_) {
    std::shared_ptr<Item> deleted_item;
    REVERB_RETURN_IF_ERROR(DeleteItem(remover_->Sample().key, &deleted_item));
    to_delete->push_back(std::move(deleted_item));
  }

  // Now that the new item has been inserted and an older item has
  // (potentially) been removed the insert can be finalized.
  rate_limiter_->Insert(&mu_);
  return absl::OkStatus();
}

absl::Status Table::MutateItems(absl::Span<const KeyWithPriority> updates,
                                absl::Span<const Key> deletes) {
  std::vector<std::shared_ptr<Item>> deleted_items(deletes.size());
  {
    absl::MutexLock lock(&mu_);
    for (int i = 0; i < deletes.size(); i++) {
      REVERB_RETURN_IF_ERROR(DeleteItem(deletes[i], &deleted_items[i]));
    }
    for (const auto& item : updates) {
      REVERB_RETURN_IF_ERROR(UpdateItem(item.key(), item.priority()));
    }
  }
  return absl::OkStatus();
}

absl::Status Table::Sample(SampledItem* sampled_item, absl::Duration timeout) {
  std::vector<SampledItem> items;
  REVERB_RETURN_IF_ERROR(SampleFlexibleBatch(&items, 1, timeout));
  *sampled_item = std::move(items[0]);
  return absl::OkStatus();
}

void Table::EnqueSampleRequest(int num_samples,
                               std::weak_ptr<SamplingCallback> callback,
                               absl::Duration timeout) {
  auto request = std::make_unique<SampleRequest>();
  request->on_batch_done = std::move(callback);
  request->deadline = absl::Now() + timeout;
  // Reserved size is used to communicate sampling batch size (it eliminates the
  // need of alocating memory inside the table worker).
  request->samples.reserve(num_samples);
  {
    absl::MutexLock lock(&worker_mu_);
    pending_sampling_.push_back(std::move(request));
    if (worker_sleeps_) {
      wakeup_worker_.Signal();
    }
  }
}

absl::Status Table::SampleFlexibleBatch(std::vector<SampledItem>* items,
                                        int batch_size,
                                        absl::Duration timeout) {
  // Allocate memory outside of critical section.
  items->reserve(batch_size);

  // Keep references to the (potentially) deleted items alive until the lock has
  // been released.
  std::vector<std::shared_ptr<Item>> deleted_items;
  {
    absl::MutexLock lock(&mu_);
    for (int i = 0; i < batch_size; i++) {
      if (auto status = rate_limiter_->AwaitAndFinalizeSample(&mu_, timeout);
          !status.ok()) {
        // Deadline exceeded errors encountered after the first call means that
        // it was not possible to proceed with another sample without awaiting
        // changes. If this happens then we simply return the items that we
        // sampled so far.
        if (i != 0 && absl::IsDeadlineExceeded(status)) {
          return absl::OkStatus();
        }
        return status;
      }

      // All calls but the first should return immediately if the rate limiter
      // does not allow for another sample call to proceed.
      timeout = absl::ZeroDuration();

      auto sample = sampler_->Sample();
      std::shared_ptr<Item>& item = data_[sample.key];

      // If this is the first time the item was sampled then
      if (item->item.times_sampled() == 0) {
        ++num_unique_samples_;
      }

      // Increment the sample count.
      item->item.set_times_sampled(item->item.times_sampled() + 1);

      // Copy Details of the sampled item.
      SampledItem sampled_item = {
          .ref = item,
          .probability = sample.probability,
          .table_size = static_cast<int64_t>(data_.size()),
          .priority = item->item.priority(),
          .times_sampled = item->item.times_sampled(),
      };
      items->push_back(std::move(sampled_item));

      // Notify extensions which item was sampled.
      for (auto& extension : extensions_) {
        extension->OnSample(&mu_, *item);
      }

      // If there is an upper bound of the number of times an item can be
      // sampled and it is now reached then delete the item before the lock is
      // released.
      if (item->item.times_sampled() == max_times_sampled_) {
        deleted_items.emplace_back();
        REVERB_RETURN_IF_ERROR(
            DeleteItem(item->item.key(), &deleted_items.back()));
      }
    }
  }

  return absl::OkStatus();
}

absl::Status Table::SampleInternal(
    SampledItem* result, std::vector<std::shared_ptr<Item>>* to_delete) {
  auto sample = sampler_->Sample();
  std::shared_ptr<Item>& item = data_[sample.key];
  // Increment the sample count.
  item->item.set_times_sampled(item->item.times_sampled() + 1);

  // Copy Details of the sampled item.
  *result = {
      .ref = item,
      .probability = sample.probability,
      .table_size = static_cast<int64_t>(data_.size()),
      .priority = item->item.priority(),
      .times_sampled = item->item.times_sampled(),
  };

  // Notify extensions which item was sampled.
  for (auto& extension : extensions_) {
    extension->OnSample(&mu_, *item);
  }

  // If there is an upper bound of the number of times an item can be
  // sampled and it is now reached then delete the item before the lock is
  // released.
  if (item->item.times_sampled() == max_times_sampled_) {
    to_delete->emplace_back();
    REVERB_RETURN_IF_ERROR(
        DeleteItem(item->item.key(), &to_delete->back()));
  }
  return absl::OkStatus();
}

int64_t Table::size() const {
  absl::MutexLock lock(&mu_);
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

  absl::MutexLock lock(&mu_);
  *info.mutable_rate_limiter_info() = rate_limiter_->Info(&mu_);
  *info.mutable_sampler_options() = sampler_->options();
  *info.mutable_remover_options() = remover_->options();
  info.set_current_size(data_.size());
  info.set_num_episodes(episode_refs_.size());
  info.set_num_deleted_episodes(num_deleted_episodes_);
  info.set_num_unique_samples(num_unique_samples_);

  return info;
}

void Table::Close() {
  {
    absl::MutexLock lock(&worker_mu_);
    stop_worker_ = true;
    wakeup_worker_.Signal();
  }
  // Join the worker thread
  table_worker_ = nullptr;
  absl::MutexLock lock(&mu_);
  rate_limiter_->Cancel(&mu_);
}

absl::Status Table::DeleteItem(Table::Key key,
                               std::shared_ptr<Item>* deleted_item) {
  auto it = data_.find(key);
  if (it == data_.end()) return absl::OkStatus();

  for (auto& extension : extensions_) {
    extension->OnDelete(&mu_, *it->second);
  }

  // Decrement counts to the episodes the item is referencing.
  for (const auto& chunk : it->second->chunks) {
    auto ep_it = episode_refs_.find(chunk->episode_id());
    REVERB_CHECK(ep_it != episode_refs_.end());
    if (--(ep_it->second) == 0) {
      episode_refs_.erase(ep_it);
      num_deleted_episodes_++;
    }
  }

  *deleted_item = std::move(it->second);
  data_.erase(it);
  rate_limiter_->Delete(&mu_);
  REVERB_RETURN_IF_ERROR(sampler_->Delete(key));
  REVERB_RETURN_IF_ERROR(remover_->Delete(key));
  return absl::OkStatus();
}

absl::Status Table::UpdateItem(
    Key key, double priority, std::initializer_list<TableExtension*> exclude) {
  auto it = data_.find(key);
  if (it == data_.end()) {
    return absl::OkStatus();
  }
  it->second->item.set_priority(priority);
  REVERB_RETURN_IF_ERROR(sampler_->Update(key, priority));
  REVERB_RETURN_IF_ERROR(remover_->Update(key, priority));

  for (auto& extension : extensions_) {
    if (std::none_of(
            exclude.begin(), exclude.end(),
            [ext_ptr = extension.get()](auto e) { return e == ext_ptr; })) {
      extension->OnUpdate(&mu_, *it->second);
    }
  }

  return absl::OkStatus();
}

absl::Status Table::Reset() {
  // Make sure worker has no more pending work.
  {
    auto worker_done = [this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(worker_mu_) {
      return worker_sleeps_ && pending_inserts_.empty();
    };
    absl::MutexLock lock(&worker_mu_);
    worker_mu_.Await(absl::Condition(&worker_done));
  }

  absl::MutexLock lock(&mu_);

  for (auto& extension : extensions_) {
    extension->OnReset(&mu_);
  }

  sampler_->Clear();
  remover_->Clear();

  num_deleted_episodes_ = 0;
  num_unique_samples_ = 0;

  data_.clear();

  rate_limiter_->Reset(&mu_);

  return absl::OkStatus();
}

Table::CheckpointAndChunks Table::Checkpoint() {
  PriorityTableCheckpoint checkpoint;
  checkpoint.set_table_name(name());
  checkpoint.set_max_size(max_size_);
  checkpoint.set_max_times_sampled(max_times_sampled_);

  if (signature_.has_value()) {
    *checkpoint.mutable_signature() = signature_.value();
  }

  absl::MutexLock lock(&mu_);

  checkpoint.set_num_deleted_episodes(num_deleted_episodes_);
  checkpoint.set_num_unique_samples(num_unique_samples_);

  *checkpoint.mutable_sampler() = sampler_->options();
  *checkpoint.mutable_remover() = remover_->options();

  // Note that is is important that the rate limiter checkpoint is
  // finalized before the items are added
  *checkpoint.mutable_rate_limiter() = rate_limiter_->CheckpointReader(&mu_);

  absl::flat_hash_set<std::shared_ptr<ChunkStore::Chunk>> chunks;
  for (const auto& entry : data_) {
    *checkpoint.add_items() = entry.second->item;
    chunks.insert(entry.second->chunks.begin(), entry.second->chunks.end());
  }

  // Sort the items in ascending order based on their insertion time. This makes
  // it possible to reconstruct ordered structures (Fifo) when the checkpoint is
  // loaded.
  std::sort(checkpoint.mutable_items()->begin(),
            checkpoint.mutable_items()->end(), IsInsertedBefore);

  return {std::move(checkpoint), std::move(chunks)};
}

absl::Status Table::InsertCheckpointItem(Table::Item item) {
  absl::MutexLock lock(&mu_);
  REVERB_CHECK_LE(data_.size() + 1, max_size_)
      << "InsertCheckpointItem called on already full Table";
  REVERB_CHECK(!data_.contains(item.item.key()))
      << "InsertCheckpointItem called for item with already present key: "
      << item.item.key();

  REVERB_RETURN_IF_ERROR(
      sampler_->Insert(item.item.key(), item.item.priority()));
  REVERB_RETURN_IF_ERROR(
      remover_->Insert(item.item.key(), item.item.priority()));

  const auto key = item.item.key();
  auto it = data_.emplace(key, std::make_shared<Item>(std::move(item))).first;
  for (auto& extension : extensions_) {
    extension->OnInsert(&mu_, *it->second);
  }

  for (const auto& chunk : it->second->chunks) {
    ++episode_refs_[chunk->episode_id()];
  }

  return absl::OkStatus();
}

bool Table::Get(Table::Key key, Table::Item* item) {
  absl::MutexLock lock(&mu_);
  auto it = data_.find(key);
  if (it != data_.end()) {
    *item = *it->second;
    return true;
  }
  return false;
}

const internal::flat_hash_map<Table::Key, std::shared_ptr<Table::Item>>*
Table::RawLookup() {
  mu_.AssertHeld();
  return &data_;
}

void Table::UnsafeAddExtension(std::shared_ptr<TableExtension> extension) {
  REVERB_CHECK_OK(extension->RegisterTable(&mu_, this));
  absl::MutexLock lock(&mu_);
  REVERB_CHECK(data_.empty());
  extensions_.push_back(std::move(extension));
}

const std::vector<std::shared_ptr<TableExtension>>& Table::extensions() const {
  return extensions_;
}

const absl::optional<tensorflow::StructuredValue>& Table::signature() const {
  return signature_;
}

bool Table::CanSample(int num_samples) const {
  absl::MutexLock lock(&mu_);
  return rate_limiter_->CanSample(&mu_, num_samples);
}

bool Table::CanInsert(int num_inserts) const {
  absl::MutexLock lock(&mu_);
  return rate_limiter_->CanInsert(&mu_, num_inserts);
}

RateLimiterEventHistory Table::GetRateLimiterEventHistory(
    size_t min_insert_event_id, size_t min_sample_event_id) const {
  absl::MutexLock lock(&mu_);
  return rate_limiter_->GetEventHistory(&mu_, min_insert_event_id,
                                        min_sample_event_id);
}

int64_t Table::num_episodes() const {
  absl::MutexLock lock(&mu_);
  return episode_refs_.size();
}

absl::Status Table::UnsafeUpdateItem(
    Key key, double priority, std::initializer_list<TableExtension*> exclude) {
  mu_.AssertHeld();
  return UpdateItem(key, priority, std::move(exclude));
}

std::vector<std::shared_ptr<TableExtension>> Table::UnsafeClearExtensions() {
  std::vector<std::shared_ptr<TableExtension>> extensions;
  {
    absl::MutexLock lock(&mu_);
    REVERB_CHECK(data_.empty());
    extensions.swap(extensions_);
  }

  for (auto& extension : extensions) {
    extension->UnregisterTable(&mu_, this);
  }

  return extensions;
}

int64_t Table::num_deleted_episodes() const {
  absl::MutexLock lock(&mu_);
  return num_deleted_episodes_;
}

void Table::set_num_deleted_episodes_from_checkpoint(int64_t value) {
  absl::MutexLock lock(&mu_);
  REVERB_CHECK(data_.empty() && num_deleted_episodes_ == 0);
  num_deleted_episodes_ = value;
}

void Table::set_num_unique_samples_from_checkpoint(int64_t value) {
  absl::MutexLock lock(&mu_);
  REVERB_CHECK(data_.empty() && num_unique_samples_ == 0);
  num_unique_samples_ = value;
}

int32_t Table::DefaultFlexibleBatchSize() const {
  const auto& rl_info = rate_limiter_->InfoWithoutCallStats();
  // When a samples per insert ratio is provided then match the batch size with
  // the ratio. Bigger values can result in worse performance when the error
  // range (min_diff -> max_diff) is small.
  if (rl_info.samples_per_insert() > 1) {
    return rl_info.samples_per_insert();
  }

  // If a min size limiter is used then we allow for big batches (64) to enable
  // the samplers to run almost completely unconstrained. If a max_times_sampled
  // is used then we reduce the batch size to avoid a single sample stream
  // consuming all the items and starving the other. Note that we check for
  // negative error buffers as very large/small max/min diffs could result in
  // overflow (which only occurs when using a min size limiter).
  double error_buffer = rl_info.max_diff() - rl_info.min_diff();
  if (rl_info.samples_per_insert() == 1 &&
      (error_buffer > max_size_ * 1000 || error_buffer < 0)) {
    return max_times_sampled_ < 1 ? 64 : max_times_sampled_;
  }

  // If all else fails, default to one sample per call.
  return 1;
}

std::string Table::DebugString() const {
  absl::MutexLock lock(&mu_);
  std::string str = absl::StrCat(
      "Table(sampler=", sampler_->DebugString(),
      ", remover=", remover_->DebugString(),
      ", max_size=", max_size_,
      ", max_times_sampled=", max_times_sampled_,
      ", name=", name_,
      ", rate_limiter=", rate_limiter_->DebugString(),
      ", signature=",
      (signature_.has_value() ? signature_.value().DebugString() : "nullptr"));

  if (!extensions_.empty()) {
    absl::StrAppend(&str, ", extensions=[");
    for (size_t i = 0; i < extensions_.size(); ++i) {
      absl::StrAppend(&str, extensions_[i]->DebugString());
      if (i != extensions_.size() - 1) {
        absl::StrAppend(&str, ", ");
      }
    }
    absl::StrAppend(&str, "]");
  }
  absl::StrAppend(&str, ")");
  return str;
}

}  // namespace reverb
}  // namespace deepmind
