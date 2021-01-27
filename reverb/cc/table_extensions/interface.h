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
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "reverb/cc/schema.pb.h"

namespace deepmind {
namespace reverb {

class Table;
class TableItem;

// A `TableExtension` is passed to a single `Table` and executed
// as part of the atomic operations of the parent table. All "hooks" are
// executed while parent is holding its mutex and thus latency is very
// important.
class TableExtension {
 public:
  virtual ~TableExtension() = default;

  // Returns a summary string description.
  virtual std::string DebugString() const = 0;

 protected:
  friend class Table;

  // Executed just after item is inserted into  parent `Table`.
  virtual void OnInsert(absl::Mutex* mu, const TableItem& item)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) = 0;

  // Executed just before item is removed from parent `Table`.
  virtual void OnDelete(absl::Mutex* mu, const TableItem& item)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) = 0;

  // Executed just after the priority of an item has been updated in parent
  // `Table`.
  virtual void OnUpdate(absl::Mutex* mu, const TableItem& item)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) = 0;

  // Executed just before a sample is returned. The sample count of the item
  // includes the active sample and thus always is >= 1.
  virtual void OnSample(absl::Mutex* mu, const TableItem& item)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) = 0;

  // Executed just before all items are deleted.
  virtual void OnReset(absl::Mutex* mu) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu) = 0;

  // Table calls these methods on construction and destruction.
  virtual absl::Status RegisterTable(absl::Mutex* mu, Table* table)
      ABSL_LOCKS_EXCLUDED(mu) = 0;
  virtual void UnregisterTable(absl::Mutex* mu, Table* table)
      ABSL_LOCKS_EXCLUDED(mu) = 0;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_TABLE_EXTENSIONS_INTERFACE_H_
