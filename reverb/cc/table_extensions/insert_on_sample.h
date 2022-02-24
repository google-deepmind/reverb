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

#ifndef REVERB_CC_TABLE_EXTENSIONS_INSERT_ON_SAMPLE_H_
#define REVERB_CC_TABLE_EXTENSIONS_INSERT_ON_SAMPLE_H_

#include "reverb/cc/table.h"
#include "reverb/cc/table_extensions/base.h"

namespace deepmind::reverb {

// Inserts an identical item into target table when the item is sampled for the
// first time from the source table (i.e. the table which owns the extension).
//
// NOTE! The item will be inserted into the target table with `times_sampled`
// set to 1 even though it has been sampled from the target table.
//
// NOTE: We assume that all inserts will succeed but if they don't then we'll
// simply drop the item and log to ERROR.
class InsertOnSampleExtension : public TableExtensionBase {
 public:
  // `ApplyOnSample` will wait at most `timeout` while inserting to the
  // `target_table`, throwing away the item to insert. absl::InfiniteDuration()
  // will block on the successful insertion.
  // If `target_table` can block on inserts, prefer using a short timeout.
  InsertOnSampleExtension(std::shared_ptr<Table> target_table,
                          absl::Duration timeout);

  // Inserts a copy of the item into the target table.
  void ApplyOnSample(const ExtensionItem& item) override;

  // Returns a summary string description.
  std::string DebugString() const override;

  bool CanRunAsync() const override { return true; }

 protected:
  void AfterRegisterTable(const Table& table) override;
  void BeforeUnregisterTable(const Table& table) override;

 private:
  std::shared_ptr<Table> target_table_;
  // We save this so that `DebugString` don't have to take the lock to access
  // `table_`.
  std::string table_name_;
  absl::Duration timeout_;
};

}  // namespace deepmind::reverb

#endif  // REVERB_CC_TABLE_EXTENSIONS_INSERT_ON_SAMPLE_H_
