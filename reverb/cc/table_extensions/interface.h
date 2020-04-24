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

  // Executed just after item is inserted into  parent `PriorityTable`.
  virtual void OnInsert(const PriorityTableItem& item) = 0;

  // Executed just before item is removed from parent `PriorityTable`.
  virtual void OnDelete(const PriorityTableItem& item) = 0;

  // Executed just after the priority of an item has been updated in parent
  // `PriorityTable`. `OnUpdate` of all registered extensions are called before
  // `Diffuse` is called.
  virtual void OnUpdate(const PriorityTableItem& item) = 0;

  // Executed just before a sample is returned. The sample count of the item
  // includes the active sample and thus always is >= 1.
  virtual void OnSample(const PriorityTableItem& item) = 0;

  // Executed just before all items are deleted.
  virtual void OnReset() = 0;

  // Diffuses the update to the neighborhood and returns a vector of updates
  // that should be applied as a result.
  //
  // `item` is the updated item after the update has been applied and
  // `old_priority` is was the priority of the item before the update was
  // applied.
  //
  // This method must only be called from `table` as mutex lock is held as part
  // of an update.
  //
  // `table` must not be nullptr and `item` must contain chunks.
  virtual std::vector<KeyWithPriority> Diffuse(PriorityTable* table,
                                               const PriorityTableItem& item,
                                               double old_priority) = 0;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_TABLE_EXTENSIONS_INTERFACE_H_
