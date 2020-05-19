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

#ifndef REVERB_CC_TABLE_EXTENSIONS_INTERFACE_H_
#define REVERB_CC_TABLE_EXTENSIONS_INTERFACE_H_

#include <vector>

#include <cstdint>
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "reverb/cc/priority_table_item.h"
#include "reverb/cc/schema.pb.h"

namespace deepmind {
namespace reverb {

class PriorityTable;

// A `PriorityTableExtension` is passed to a single `PriorityTable` and executed
// as part of the atomic operations of the parent table. All "hooks" are
// executed while parent is holding its mutex and thus latency is very
// important.
class PriorityTableExtensionInterface {
 public:
  virtual ~PriorityTableExtensionInterface() = default;

 protected:
  friend class PriorityTable;

  // Executed just after item is inserted into  parent `PriorityTable`.
  virtual void OnInsert(absl::Mutex* mu, const PriorityTableItem& item)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) = 0;

  // Executed just before item is removed from parent `PriorityTable`.
  virtual void OnDelete(absl::Mutex* mu, const PriorityTableItem& item)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) = 0;

  // Executed just after the priority of an item has been updated in parent
  // `PriorityTable`.
  virtual void OnUpdate(absl::Mutex* mu, const PriorityTableItem& item)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) = 0;

  // Executed just before a sample is returned. The sample count of the item
  // includes the active sample and thus always is >= 1.
  virtual void OnSample(absl::Mutex* mu, const PriorityTableItem& item)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) = 0;

  // Executed just before all items are deleted.
  virtual void OnReset(absl::Mutex* mu) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) = 0;

  // PriorityTable calls these methods on construction and destruction.
  virtual tensorflow::Status RegisterPriorityTable(PriorityTable* table) = 0;
  virtual void UnregisterPriorityTable(absl::Mutex* mu, PriorityTable* table)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) = 0;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_TABLE_EXTENSIONS_INTERFACE_H_
