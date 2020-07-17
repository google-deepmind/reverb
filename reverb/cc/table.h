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

#ifndef REVERB_CC_TABLE_H_
#define REVERB_CC_TABLE_H_

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cstdint>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/interface.h"
#include "reverb/cc/table_extensions/interface.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace deepmind {
namespace reverb {

// Used for representing items of the priority distribution. See
// PrioritizedItem in schema.proto for documentation.
struct TableItem {
  PrioritizedItem item;
  std::vector<std::shared_ptr<ChunkStore::Chunk>> chunks;
};

// A `Table` is a structure for storing `TableItem` objects. The table uses two
// instances of `ItemSelectorInterface`, one for sampling (`sampler`) and
// another for removing (`remover`). All item operations (insert/update/delete)
// on the table are propagated to the sampler and remover with the original
// operation on the table. The `Table` uses the sampler to determine which items
// it should return when `Table::Sample()` is called. Similarly, the remover is
// used to determine which items should be deleted to ensure capacity.
//
// A `RateLimiter` is used to set the ratio of inserted to sampled
// items. This means that calls to `Table::InsertOrAssign()` and
// `Table::Sample()` may be blocked by the `RateLimiter` as it enforces this
// ratio.
//
// Please note that the remover is only used to limit the number of items in
// the table, not the number of data elements nor the memory used. Each item
// references one or more chunks, each chunk holds one or more data elements and
// consumes and "unknown" amount of memory. Each chunk can be referenced by any
// number of items across all tables on the server. Deleting a single item from
// one table simply decrements the reference count of the chunks it references
// and only when a chunk is referenced by zero items is it destroyed and its
// memory deallocated.
//
// This means you must be careful when choosing the remover strategy. A
// dangerous example would be using a FIFO  remover for one table and then
// introducing another with table with a  LIFO remover. In this scenario, the
// two tables would not share any chunks and would this require twice the
// amount of memory compared to two tables with the same type of remover.
//
class Table {
 public:
  using Key = ItemSelectorInterface::Key;
  using Item = TableItem;

  // Used as the return of Sample(). Note that this returns the probability of
  // an item instead as opposed to the raw priority value.
  struct SampledItem {
    PrioritizedItem item;
    std::vector<std::shared_ptr<ChunkStore::Chunk>> chunks;
    double probability;
    int64_t table_size;
  };

  // Used when checkpointing to ensure that none of the chunks referenced by the
  // checkpointed items are removed before the checkpoint operations has
  // completed.
  struct CheckpointAndChunks {
    PriorityTableCheckpoint checkpoint;
    absl::flat_hash_set<std::shared_ptr<ChunkStore::Chunk>> chunks;
  };

  // Constructor.
  // `name` is the name of the table. Must be unique within server.
  // `sampler` is used in Sample() calls, while `remover` is used in
  //   InsertOrAssign() when we need to remove an item to not exceed `max_size`
  //   items in this container.
  // `max_times_sampled` is the maximum number of times we allow for an item to
  //   be sampled before it is deleted. No value lower than 1 will be used.
  // `rate_limiter` controls when sample and insert calls are allowed to
  //   proceed.
  // `extensions` allows additional features to be injected into the table.
  // `signature` allows an optional declaration of the data that can be stored
  //   in this table.  writers and readers are responsible for checking against
  //   this signature, as it is available via RPC request.
  Table(std::string name, std::shared_ptr<ItemSelectorInterface> sampler,
        std::shared_ptr<ItemSelectorInterface> remover, int64_t max_size,
        int32_t max_times_sampled, std::shared_ptr<RateLimiter> rate_limiter,
        std::vector<std::shared_ptr<TableExtensionInterface>> extensions = {},
        absl::optional<tensorflow::StructuredValue> signature = absl::nullopt);

  ~Table();

  // Copies at most `count` items that are currently in the table.
  // If `count` is `0` (default) then all items are copied.
  // If `count` is less than `size` then a subset is selected with in an
  // undefined manner.
  std::vector<Item> Copy(size_t count = 0) const;

  // Attempts to insert an item into the distribution. If the item
  // already exists, the existing item is updated. Also applies the necessary
  // updates to sampler and remover.
  //
  // This call also ensures that the container does not grow larger than
  // `max_size`. If an insertion causes the container to exceed `max_size_`, one
  // item is removed with the strategy specified by the `remover_`. Please note
  // that we insert the new item that exceeds the capacity BEFORE we run the
  // remover. This means that the newly inserted item could be deleted right
  // away.
  tensorflow::Status InsertOrAssign(Item item);

  // Inserts an item without consulting or modifying the RateLimiter about the
  // operation.
  //
  // This should ONLY be used when restoring a `Table` from a checkpoint.
  tensorflow::Status InsertCheckpointItem(Item item);

  // Updates the priority or deletes items in this table distribution. All
  // operations in the arguments are applied in the order that they are listed.
  // Different operations can be set at the same time. Ignores non existing keys
  // but returns any other errors. The operations might be applied partially
  // when an error occurs.
  tensorflow::Status MutateItems(absl::Span<const KeyWithPriority> updates,
                                 absl::Span<const Key> deletes);

  // Attempts to sample an item from table with the sampling
  // strategy passed to the constructor. We only allow the sample operation if
  // the `rate_limiter_` allows it. If the item has reached
  // `max_times_sampled_`, then we delete it before returning so it cannot be
  // sampled again.  If `Sample` waits for `rate_limiter_` for longer than
  // `timeout`, instead of sampling a `DeadlineExceeded` status is returned.
  tensorflow::Status Sample(SampledItem* item,
                            absl::Duration timeout = kDefaultTimeout);

  // Attempts to sample up to `batch_size` items (without releasing the lock).
  //
  // The behaviour is as follows:
  //
  //   1. Block (for at most `timeout`) until `rate_limiter_` allows (at least)
  //      one sample operation to proceed. At this point an exclusive lock on
  //      the table is acquired.
  //   2. If `timeout` was exceeded, return `DeadlineExceededError`.
  //   3. Select item using `sampler_`, push item to output vector `items`,
  //      call extensions and delete item from table if `max_times_sampled_`
  //      reached.
  //   4. (Without releasing the lock) IFF `rate_limiter_` allows for one more
  //      sample operation to proceed AND `items->size() < batch_size` then go
  //      to 3, otherwise return OK.
  //
  // Note that the timeout is ONLY used when waiting for the first sample
  // operation to be "approved" by the rate limiter. The remaining items of the
  // batch will only be added if these can proceeed without releasing the lock
  // and awaiting state changes in the rate limiter.
  tensorflow::Status SampleFlexibleBatch(
      std::vector<SampledItem>* items, int batch_size,
      absl::Duration timeout = kDefaultTimeout);

  // Returns true iff the current state would allow for `num_samples` to be
  // sampled. Dies if `num_samples` is < 1.
  //
  // TODO(b/153258711): This currently ignores max_size and
  // max_times_sampled arguments to the table, and will return true if e.g.
  // there are 2 items in the table, max_times_sampled=1, and num_samples=3.
  bool CanSample(int num_samples) const;

  // Returns true iff the current state would allow for `num_inserts` to be
  // inserted. Dies if `num_inserts` is < 1.
  //
  // TODO(b/153258711): This currently ignores max_size and max_times_sampled
  // arguments to the table.
  bool CanInsert(int num_inserts) const;

  // Appends the extension to the internal list. Note that this must be called
  // before any other operation is called. If called when the number of items
  // is non zero, death is triggered.
  //
  // Note! This method is not thread safe and caller is responsible for making
  // sure that this method, nor any other method, is called concurrently.
  void UnsafeAddExtension(std::shared_ptr<TableExtensionInterface> extension);

  // Unregisters and returns all extension from the internal list. Note that
  // this must be called before items are inserted. If called when the number of
  // items is non zero, death is triggered.
  //
  // Note! This method is not thread safe and caller is responsible for making
  // sure that this method, nor any other method, is called concurrently.
  std::vector<std::shared_ptr<TableExtensionInterface>> UnsafeClearExtensions();

  // Registered table extensions.
  const std::vector<std::shared_ptr<TableExtensionInterface>>& extensions()
      const;

  // Lookup a single item. Returns true if found, else false.
  bool Get(Key key, Item* item) ABSL_LOCKS_EXCLUDED(mu_);

  // Get pointer to `data_`. Must only be called by extensions while lock held.
  const absl::flat_hash_map<Key, Item>* RawLookup()
      ABSL_ASSERT_SHARED_LOCK(mu_);

  // Removes all items and resets the RateLimiter to its initial state.
  tensorflow::Status Reset();

  // Generate a checkpoint from the table's current state.
  CheckpointAndChunks Checkpoint();

  // Number of items in the table distribution.
  int64_t size() const;

  // Number of episodes in the table.
  int64_t num_episodes() const ABSL_LOCKS_EXCLUDED(mu_);

  // Number of episodes that previously were in the table but has since been
  // deleted.
  int64_t num_deleted_episodes() const ABSL_LOCKS_EXCLUDED(mu_);

  // "Manually" set the number of deleted episodes. This is only intended to be
  // called when reconstructing a Table from a checkpoint and will trigger death
  // unless it is the very first interaction with the table.
  void set_num_deleted_episodes_from_checkpoint(int64_t value)
      ABSL_LOCKS_EXCLUDED(mu_);

  const std::string& name() const;

  // Metadata about the table, including the current state of the rate limiter.
  TableInfo info() const;

  // Signature (if any) of the table.
  const absl::optional<tensorflow::StructuredValue>& signature() const;

  // Makes a copy of all COMPLETED rate limiter events since (inclusive)
  // `min_X_event_id`.
  RateLimiterEventHistory GetRateLimiterEventHistory(
      size_t min_insert_event_id, size_t min_sample_event_id) const
      ABSL_LOCKS_EXCLUDED(mu_);

  // Cancels pending calls and marks object as closed. Object must be
  // abandoned after `Close` called.
  void Close();

  // Asserts that `mu_` is held at runtime and calls UpdateItem.
  tensorflow::Status UnsafeUpdateItem(
      Key key, double priority,
      std::initializer_list<TableExtensionInterface*> exclude)
      ABSL_ASSERT_EXCLUSIVE_LOCK(mu_);

 private:
  // Updates item priority in `data_`, `samper_`, `remover_` and calls
  // `OnUpdate` on all extensions not part of `exclude`.
  tensorflow::Status UpdateItem(
      Key key, double priority,
      std::initializer_list<TableExtensionInterface*> exclude = {})
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Deletes the item associated with the key from `data_`, `sampler_` and
  // `remover_`. Ignores the key if it cannot be found.
  //
  // The deleted item is returned in order to allow the deallocation of the
  // underlying item to be postponed until the lock has been released.
  tensorflow::Status DeleteItem(Key key, Item* deleted_item)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Distribution used for sampling.
  std::shared_ptr<ItemSelectorInterface> sampler_ ABSL_GUARDED_BY(mu_);

  // Distribution used for removing.
  std::shared_ptr<ItemSelectorInterface> remover_ ABSL_GUARDED_BY(mu_);

  // Bijection of key to item. Used for storing the chunks and timestep range of
  // each item.
  absl::flat_hash_map<Key, Item> data_ ABSL_GUARDED_BY(mu_);

  // Count of references from chunks referenced by items.
  absl::flat_hash_map<uint64_t, int64_t> episode_refs_ ABSL_GUARDED_BY(mu_);

  // The total number of episodes that were at some point referenced by items
  // in the table but have since been removed. Is set to 0 when `Reset()`
  // called.
  int64_t num_deleted_episodes_ ABSL_GUARDED_BY(mu_);

  // Maximum number of items that this container can hold. InsertOrAssign()
  // respects this limit when inserting a new item.
  const int64_t max_size_;

  // Maximum number of times an item can be sampled before it is deleted.
  // A value <= 0 means there is no limit.
  const int32_t max_times_sampled_;

  // Name of the table.
  const std::string name_;

  // Controls what operations can proceed. A shared_ptr is used to allow the
  // Python layer to interact with the object after it has been passed to the
  // table.
  std::shared_ptr<RateLimiter> rate_limiter_ ABSL_GUARDED_BY(mu_);

  // Extensions implement hooks that are executed while holding `mu_` as part
  // of insert, delete, update or reset operations.
  std::vector<std::shared_ptr<TableExtensionInterface>> extensions_
      ABSL_GUARDED_BY(mu_);

  // Synchronizes access to `sampler_`, `remover_`, 'rate_limiter_`,
  // 'extensions_` and `data_`,
  mutable absl::Mutex mu_;

  // Optional signature for data in the table.
  const absl::optional<tensorflow::StructuredValue> signature_;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_TABLE_H_
