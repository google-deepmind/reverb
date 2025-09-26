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

#include "reverb/cc/table_extensions/base.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/table.h"

namespace deepmind {
namespace reverb {

absl::Status TableExtensionBase::RegisterTable(absl::Mutex* mu, Table* table) {
  absl::MutexLock lock(table_mu_);
  if (table_) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Attempting to registering a table ", absl::Hex(table),
        " (name: ", table->name(), ") with extension that has already been ",
        "registered with: ", absl::Hex(table_), " (name: ", table->name(),
        ")"));
  }
  table_ = table;
  AfterRegisterTable(*table_);
  return absl::OkStatus();
}

void TableExtensionBase::UnregisterTable(absl::Mutex* mu, Table* table) {
  absl::MutexLock lock(table_mu_);
  REVERB_CHECK_EQ(table, table_)
      << "The wrong Table attempted to unregister this extension.";
  BeforeUnregisterTable(*table_);
  table_ = nullptr;
}

void TableExtensionBase::OnDelete(absl::Mutex* mu, const ExtensionItem& item) {
  ApplyOnDelete(item);
}

void TableExtensionBase::OnInsert(absl::Mutex* mu, const ExtensionItem& item) {
  ApplyOnInsert(item);
}

void TableExtensionBase::OnReset(absl::Mutex* mu) { ApplyOnReset(); }

void TableExtensionBase::OnUpdate(absl::Mutex* mu, const ExtensionItem& item) {
  ApplyOnUpdate(item);
}

void TableExtensionBase::OnSample(absl::Mutex* mu, const ExtensionItem& item) {
  ApplyOnSample(item);
}

void TableExtensionBase::ApplyOnDelete(const ExtensionItem& item) {}

void TableExtensionBase::ApplyOnInsert(const ExtensionItem& item) {}

void TableExtensionBase::ApplyOnReset() {}

void TableExtensionBase::ApplyOnUpdate(const ExtensionItem& item) {}

void TableExtensionBase::ApplyOnSample(const ExtensionItem& item) {}

}  // namespace reverb
}  // namespace deepmind
