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
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/chunk_store.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/interface.h"
#include "reverb/cc/support/task_executor.h"
#include "reverb/cc/table_extensions/interface.h"
#include "tensorflow/core/protobuf/struct.pb.h"

namespace deepmind {
namespace reverb {

// Used for representing items of the priority distribution. See
// PrioritizedItem in schema.proto for documentation.
struct TableItem {
  PrioritizedItem item;
  std::vector<std::shared_ptr<ChunkStore::Chunk>> chunks;
};

// Table item wrapper used by extensions. It holds shared pointer to the
// TableItem to avoid copies, while also holds priority and times_sampled, which
// are mutable part of the TableItem.
struct ExtensionItem {
  ExtensionItem(std::shared_ptr<TableItem> item) : ref(std::move(item)) {
    times_sampled = ref->item.times_sampled();
    priority = ref->item.priority();
  }
  std::shared_ptr<TableItem> ref;
  int32_t times_sampled;
  double priority;
};

// A `Table` is a structure for storing `TableItem` objects. The table uses two
// instances of `ItemSelector`, one for sampling (`sampler`) and
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
  // Maximum number of enqueued inserts that are allowed on the table without
  // slowing down further inserts:
  // - absolute value limit.
  // - table's maximum size percentage limit.
  static constexpr int64_t kMaxEnqueuedInserts = 1000;
  static constexpr float kMaxEnqueuedInsertsPerc = 0.1;

  // Maximum number of allowed enqueued extension operations.
  // - absolute value limit.
  // - table's maximum size percentage limit.
  static constexpr int64_t kMaxPendingExtensionOps = 1000;
  static constexpr float kMaxPendingExtensionOpsPerc = 0.1;

  struct SampleRequest;
  using Key = ItemSelector::Key;
  using Item = TableItem;
  using SamplingCallback = std::function<void(SampleRequest*)>;

  // Used as the return of Sample(). Note that this returns the probability of
  // an item instead as opposed to the raw priority value.
  struct SampledItem {
    std::shared_ptr<Item> ref;
    double probability;
    int64_t table_size;
    // Use these values over accessing priority and times_sampled from the
    // referenced Item, as Item might be modified in the background.
    double priority;
    int32_t times_sampled;
    // True if the sample was delayed due to rate limiting. That is, the system
    // stopped proccessing requests even though there were outstanding sample
    // requests to be fulfilled.
    bool rate_limited;
  };

  // Represents asynchronous sampling request processed by the table worker.
  struct SampleRequest {
    std::vector<SampledItem> samples;
    absl::Time deadline;
    absl::Status status;
    std::weak_ptr<SamplingCallback> on_batch_done;
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
  Table(std::string name, std::shared_ptr<ItemSelector> sampler,
        std::shared_ptr<ItemSelector> remover, int64_t max_size,
        int32_t max_times_sampled, std::shared_ptr<RateLimiter> rate_limiter,
        std::vector<std::shared_ptr<TableExtension>> extensions = {},
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
  //
  // The timeout is forwarded to the rate limiter. If the right to insert cannot
  // be acquired before the timeout is exceeded the `DeadlineExceededError` is
  // returned and no action is taken.
  absl::Status InsertOrAssign(
      Item item, absl::Duration timeout = absl::InfiniteDuration());

  // Similar to the InsertOrAssign, but insert operation is queued inside the
  // table instead of blocking the caller. can_insert_more is set to true if
  // further inserts can be performed right away. When can_insert_more is set
  // to false, insert_more_callback callback will be called when further inserts
  // are allowed (with OK status), or with an error in case system is being
  // terminated.
  absl::Status InsertOrAssignAsync(
      Item item, bool* can_insert_more,
      std::weak_ptr<std::function<void(const absl::Status&)>>
          insert_more_callback);

  // Inserts an item without consulting or modifying the RateLimiter about the
  // operation.
  //
  // This should ONLY be used when restoring a `Table` from a checkpoint.
  absl::Status InsertCheckpointItem(Item item);

  // Updates the priority or deletes items in this table distribution. All
  // operations in the arguments are applied in the order that they are listed.
  // Different operations can be set at the same time. Ignores non existing keys
  // but returns any other errors. The operations might be applied partially
  // when an error occurs.
  absl::Status MutateItems(absl::Span<const KeyWithPriority> updates,
                           absl::Span<const Key> deletes);

  // Attempts to sample an item from table with the sampling
  // strategy passed to the constructor. We only allow the sample operation if
  // the `rate_limiter_` allows it. If the item has reached
  // `max_times_sampled_`, then we delete it before returning so it cannot be
  // sampled again.  If `Sample` waits for `rate_limiter_` for longer than
  // `timeout`, instead of sampling a `DeadlineExceeded` status is returned.
  absl::Status Sample(SampledItem* item,
                      absl::Duration timeout = kDefaultTimeout);

  // Enques an asynchronous sampling performed by the table worker.
  // All queued sampling operations are a subject to the table's sampling
  // strategy defined by the `rate_limiter_`. Sampled element which has reached
  // `max_times_sampled_` are deleted from the table, so it cannot be
  // sampled again.
  void EnqueSampleRequest(int num_samples,
                          std::weak_ptr<SamplingCallback> callback,
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
  absl::Status SampleFlexibleBatch(std::vector<SampledItem>* items,
                                   int batch_size,
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
  void UnsafeAddExtension(std::shared_ptr<TableExtension> extension);

  // Unregisters and returns all extension from the internal list. Note that
  // this must be called before items are inserted. If called when the number of
  // items is non zero, death is triggered.
  //
  // Note! This method is not thread safe and caller is responsible for making
  // sure that this method, nor any other method, is called concurrently.
  std::vector<std::shared_ptr<TableExtension>> UnsafeClearExtensions();

  // Lookup a single item. Returns true if found, else false.
  bool Get(Key key, Item* item) ABSL_LOCKS_EXCLUDED(mu_);

  // Get pointer to `data_`. Must only be called by extensions while lock held.
  const internal::flat_hash_map<Key, std::shared_ptr<Item>>* RawLookup()
      ABSL_ASSERT_EXCLUSIVE_LOCK(mu_);

  // Removes all items and resets the RateLimiter to its initial state.
  absl::Status Reset();

  // Generate a checkpoint from the table's current state.
  CheckpointAndChunks Checkpoint() ABSL_LOCKS_EXCLUDED(mu_);

  // Number of items in the table distribution.
  int64_t size() const ABSL_LOCKS_EXCLUDED(mu_);

  // Number of episodes in the table.
  int64_t num_episodes() const ABSL_LOCKS_EXCLUDED(mu_);

  // Number of episodes that previously were in the table but has since been
  // deleted.
  int64_t num_deleted_episodes() const ABSL_LOCKS_EXCLUDED(mu_);

  // "Manually" set the number of deleted episodes and unique samples. This is
  // only intended to be called when reconstructing a Table from a checkpoint
  // and will trigger death unless it is the very first interaction with the
  // table.
  void set_num_deleted_episodes_from_checkpoint(int64_t value)
      ABSL_LOCKS_EXCLUDED(mu_);
  void set_num_unique_samples_from_checkpoint(int64_t value)
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
  absl::Status UnsafeUpdateItem(Key key, double priority)
      ABSL_ASSERT_EXCLUSIVE_LOCK(mu_);

  // Suggestion of default batch size to use in `SampleFlexibleBatch`.
  int32_t DefaultFlexibleBatchSize() const;

  // Returns a summary string description.
  std::string DebugString() const;

  // Make the table use table worker for executing operations. Worker will use
  // provided executor for running operation callbacks. This method should be
  // called before adding any data to the table.
  void EnableTableWorker(std::shared_ptr<TaskExecutor> executor);

  // Check whether the worker is currently sleeping. This method is only exposed
  // for testing purposes.
  bool worker_is_sleeping() const ABSL_LOCKS_EXCLUDED(worker_mu_);

  // Get the number of sample requests which hasn't been picked up by the worker
  // yet. This method is only exposed for testing purposes.
  int num_pending_async_sample_requests() const ABSL_LOCKS_EXCLUDED(worker_mu_);

  // Number of queued inserts that are allowed on the table without slowing down
  // further inserts.
  int max_enqueued_inserts() const { return max_enqueued_inserts_; }

 private:
  // State of the table worker.
  enum class TableWorkerState {
    // Worker is actively processing requests.
    kRunning,

    // Worker is sleeping as there is no work to do.
    kSleeping,

    // Worker is blocked on rate limiter.
    kBlocked,
  };

  struct ExtensionRequest {
    enum class CallType { kDelete, kInsert, kSample, kUpdate, kMemoryRelease };
    CallType call_type;
    ExtensionItem item;
  };

  // Table worker execution loop. It is executed by a dedicated thread
  // and performs enqueued table operations (inserts, mutations, sampling...).
  absl::Status TableWorkerLoop();

  // Updates item priority in `data_`, `samper_`, `remover_` and calls
  // `OnUpdate` on all extensions.
  absl::Status UpdateItem(Key key, double priority)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Used by the table worker to perform sampling.
  absl::Status SampleInternal(bool rate_limited, SampledItem* result)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Finalize sampling request with a given status.
  void FinalizeSampleRequest(std::unique_ptr<Table::SampleRequest> request,
                             absl::Status status);

  // Performs insertion of the `item` into the table.
  absl::Status InsertOrAssignInternal(std::shared_ptr<Item> item)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Deletes the item associated with the key from `data_`, `sampler_` and
  // `remover_`. Ignores the key if it cannot be found.
  //
  // The deleted item is returned in order to allow the deallocation of the
  // underlying item to be postponed until the lock has been released.
  // If deleted_item is not provided, deletion is handled by the
  // extension worker asynchronously.
  absl::Status DeleteItem(Key key,
                          std::shared_ptr<Item>* deleted_item = nullptr)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Executes a given extension operation for all extensions registered with the
  // table. If extension worker is enabled, operation is executed asynchronously
  // for all extensions that support asynchronous execution. For synchronous
  // extensions operation is executed synchronously.
  void ExtensionOperation(ExtensionRequest::CallType type,
                          const std::shared_ptr<Item>& item)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Extensions worker execution loop. It is executed by a dedicated thread
  // and it performs enqueued extension operations. Operations are executed in
  // the order enqueued, but they run without holding table's lock.
  absl::Status ExtensionsWorkerLoop();

  // Synchronizes access to the table's data. Needs to be acquired to sample or
  // insert data into the table. Synchronous extensions are also executed while
  // holding this mutex.
  mutable absl::Mutex mu_ ABSL_ACQUIRED_AFTER(worker_mu_);

  // Distribution used for sampling.
  std::shared_ptr<ItemSelector> sampler_ ABSL_GUARDED_BY(mu_);

  // Distribution used for removing.
  std::shared_ptr<ItemSelector> remover_ ABSL_GUARDED_BY(mu_);

  // Bijection of key to item. Used for storing the chunks and timestep range of
  // each item.
  internal::flat_hash_map<Key, std::shared_ptr<Item>> data_
      ABSL_GUARDED_BY(mu_);

  // Count of references from chunks referenced by items.
  internal::flat_hash_map<uint64_t, int64_t> episode_refs_ ABSL_GUARDED_BY(mu_);

  // The total number of episodes that were at some point referenced by items
  // in the table but have since been removed. Is set to 0 when `Reset()`
  // called.
  int64_t num_deleted_episodes_ ABSL_GUARDED_BY(mu_);

  // The total number of unique items sampled from the table since the table
  // was created or reset most recently.
  int64_t num_unique_samples_ ABSL_GUARDED_BY(mu_);

  // Maximum number of items that this container can hold. InsertOrAssign()
  // respects this limit when inserting a new item.
  const int64_t max_size_;

  // Number of queued inserts that are allowed on the table without slowing down
  // further inserts.
  const int64_t max_enqueued_inserts_;

  // Maximum number of allowed enqueued extension operations.
  const int64_t max_enqueued_extension_ops_;

  // Maximum number of times an item can be sampled before it is deleted.
  // A value <= 0 means there is no limit.
  const int32_t max_times_sampled_;

  // Name of the table.
  const std::string name_;

  // Controls what operations can proceed. A shared_ptr is used to allow the
  // Python layer to interact with the object after it has been passed to the
  // table.
  std::shared_ptr<RateLimiter> rate_limiter_;

  // Optional signature for data in the table.
  const absl::optional<tensorflow::StructuredValue> signature_;

  // Worker thread which processes asynchronous insert and sample requests.
  std::unique_ptr<internal::Thread> table_worker_;

  // Pending asynchronous insert requests to the table.
  std::vector<std::shared_ptr<Item>> pending_inserts_
      ABSL_GUARDED_BY(worker_mu_);

  // Pending sample requests to the table (not yet picked up by the worker).
  std::vector<std::unique_ptr<SampleRequest>> pending_sampling_
      ABSL_GUARDED_BY(worker_mu_);

  // Items collected by the worker for asynchronous deletion by the clients.
  // This way we avoid expensive memory dealocation inside the worker.
  std::vector<std::shared_ptr<Item>> deleted_items_ ABSL_GUARDED_BY(worker_mu_);

  // Callbacks to be executed when further asynchronous insertions into the
  // table are allowed. Callbacks are provided with asynchronous inserts
  // and only pushed to this list if more inserts were not possible straight
  // away.
  std::vector<std::weak_ptr<std::function<void(const absl::Status&)>>>
      notify_inserts_ok_ ABSL_GUARDED_BY(worker_mu_);

  // Current table worker's state.
  TableWorkerState worker_state_ ABSL_GUARDED_BY(worker_mu_) =
      TableWorkerState::kSleeping;

  // Should worker terminate. Set to true upon table termination to stop the
  // worker.
  bool stop_worker_ ABSL_GUARDED_BY(worker_mu_) = false;

  // Used for waking up a table worker when asleep.
  absl::CondVar wakeup_worker_ ABSL_GUARDED_BY(worker_mu_);

  // Mutex to protect table worker's state.
  mutable absl::Mutex worker_mu_ ABSL_ACQUIRED_BEFORE(mu_);

  // Executor used by the table worker to run operation callbacks.
  std::shared_ptr<TaskExecutor> callback_executor_;

  // Extension worker which asynchronously updates monitoring.
  std::unique_ptr<internal::Thread> extension_worker_;

  // Pending extension requests to be processed by the extension worker.
  std::vector<ExtensionRequest> extension_requests_ ABSL_ACQUIRED_BEFORE(mu_);

  // Used for waking up extension worker when asleep.
  absl::CondVar extension_work_available_cv_ ABSL_GUARDED_BY(mu_);

  // Used for waking up table worker when space to add more extension requests
  // is available.
  absl::CondVar extension_buffer_available_cv_ ABSL_GUARDED_BY(mu_);
  bool extension_worker_sleeps_ ABSL_GUARDED_BY(mu_) = true;

  // Should extension worker terminate. Set to true upon table termination to
  // stop the worker.
  bool stop_extension_worker_ ABSL_GUARDED_BY(mu_) = false;

  // Extensions implement hooks that are executed as part of insert, delete,
  // update or reset operations. There are two types of extensions supported:
  //   - synchronous, which run while holding table's `mu_` mutex.
  //   - asynchronous, which are executed asynchronously by the extension
  //     executor. Table's mutex is not held.
  std::vector<std::shared_ptr<TableExtension>> sync_extensions_
      ABSL_GUARDED_BY(mu_);

  // Are there async extensions to run asynchronously. Used to avoid grabbing
  // async_extensions_mu_.
  bool has_async_extensions_ ABSL_GUARDED_BY(mu_) = false;

  // Mutex which has to be held to operate on asynchronous extensions.
  mutable absl::Mutex async_extensions_mu_ ABSL_ACQUIRED_AFTER(mu_);

  // Collection of asynchronous extensions. It is populated only when extension
  // worker is enabled.
  std::vector<std::shared_ptr<TableExtension>> async_extensions_
      ABSL_GUARDED_BY(async_extensions_mu_);
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_TABLE_H_
