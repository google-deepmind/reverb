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

#ifndef REVERB_CC_SELECTORS_HEAP_H_
#define REVERB_CC_SELECTORS_HEAP_H_

#include <cstdint>
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/selectors/interface.h"
#include "reverb/cc/support/intrusive_heap.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {

// HeapSelector always samples the item with the lowest or highest priority
// (controlled by `min_heap`). If multiple items share the same priority then
// the least recently inserted or updated key is sampled.
class HeapSelector : public ItemSelector {
 public:
  explicit HeapSelector(bool min_heap = true);

  // O(log n) time.
  tensorflow::Status Delete(Key key) override;

  // O(log n) time.
  tensorflow::Status Insert(Key key, double priority) override;

  // O(log n) time.
  tensorflow::Status Update(Key key, double priority) override;

  // O(1) time.
  KeyWithProbability Sample() override;

  // O(n) time.
  void Clear() override;

  KeyDistributionOptions options() const override;

  std::string DebugString() const override;

 private:
  struct HeapNode {
    Key key;
    double priority;
    IntrusiveHeapLink heap;
    uint64_t update_number;

    HeapNode(Key key, double priority, uint64_t update_number)
        : key(key), priority(priority), update_number(update_number) {}
  };

  struct HeapNodeCompare {
    bool operator()(const HeapNode* a, const HeapNode* b) const {
      // Lexicographic ordering by (priority, update_number).
      return (a->priority < b->priority) ||
             ((a->priority == b->priority) &&
              (a->update_number < b->update_number));
    }
  };

  // 1 if `min_heap` = true, else -1. Priorities are multiplied by this number
  // to control whether the min or max priority item should be sampled.
  const double sign_;

  // Heap where the top item is the one with the lowest/highest priority in the
  // distribution.
  IntrusiveHeap<HeapNode, HeapNodeCompare> heap_;

  // `IntrusiveHeap` does not manage the memory of its nodes so they are stored
  // in `nodes_`. The content of nodes_ and heap_ are always kept in sync.
  internal::flat_hash_map<Key, std::unique_ptr<HeapNode>> nodes_;

  // Keep track of the number of inserts/updates for most-recent tie-breaking.
  uint64_t update_count_;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SELECTORS_HEAP_H_
