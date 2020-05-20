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

#ifndef REVERB_CC_TABLE_EXTENSIONS_BASE_H_
#define REVERB_CC_TABLE_EXTENSIONS_BASE_H_

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "reverb/cc/priority_table_item.h"
#include "reverb/cc/table.h"
#include "reverb/cc/table_extensions/interface.h"

namespace deepmind {
namespace reverb {

// Base implementation for PriorityTableExtensionInterface.
//
// This class implements table registration and all mutex protected On*-methods
// by delegating it to a "simpler" ApplyOn method. Children are thus able to
// implement any subset of the ApplyOn (and avoid the overly verbose API)
// without losing the safety provided by the static analysis of the mutexes.
//
class PriorityTableExtensionBase : public PriorityTableExtensionInterface {
 public:
  virtual ~PriorityTableExtensionBase() = default;

  // Children should override these (noop by default).
  virtual void ApplyOnDelete(const PriorityTableItem& item);
  virtual void ApplyOnInsert(const PriorityTableItem& item);
  virtual void ApplyOnReset();
  virtual void ApplyOnUpdate(const PriorityTableItem& item);
  virtual void ApplyOnSample(const PriorityTableItem& item);

 protected:
  friend class Table;

  // Validates table and saves it to table_.
  tensorflow::Status RegisterTable(absl::Mutex* mu, Table* table)
      ABSL_LOCKS_EXCLUDED(mu) override;
  void UnregisterTable(absl::Mutex* mu, Table* table)
      ABSL_LOCKS_EXCLUDED(mu) override;

  // Delegates call to ApplyOnDelete.
  void OnDelete(absl::Mutex* mu, const PriorityTableItem& item) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Delegates call to ApplyOnInsert.
  void OnInsert(absl::Mutex* mu, const PriorityTableItem& item) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Delegates call to ApplyOnReset.
  void OnReset(absl::Mutex* mu) override ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Delegates call to ApplyOnUpdate.
  void OnUpdate(absl::Mutex* mu, const PriorityTableItem& item) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

  // Delegates call to ApplyOnSample.
  void OnSample(absl::Mutex* mu, const PriorityTableItem& item) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu);

 protected:
  absl::Mutex table_mu_;
  Table* table_ ABSL_GUARDED_BY(table_mu_) = nullptr;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_TABLE_EXTENSIONS_BASE_H_
