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

#include "reverb/cc/selectors/fifo.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/interface.h"

namespace deepmind {
namespace reverb {

absl::Status FifoSelector::Delete(ItemSelector::Key key) {
  auto it = key_to_iterator_.find(key);
  if (it == key_to_iterator_.end())
    return absl::InvalidArgumentError(absl::StrCat("Key ", key, " not found."));
  keys_.erase(it->second);
  key_to_iterator_.erase(it);
  return absl::OkStatus();
}

absl::Status FifoSelector::Insert(ItemSelector::Key key, double priority) {
  if (key_to_iterator_.find(key) != key_to_iterator_.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Key ", key, " already inserted."));
  }
  key_to_iterator_.emplace(key, keys_.emplace(keys_.end(), key));
  return absl::OkStatus();
}

absl::Status FifoSelector::Update(ItemSelector::Key key, double priority) {
  if (key_to_iterator_.find(key) == key_to_iterator_.end()) {
    return absl::InvalidArgumentError(absl::StrCat("Key ", key, " not found."));
  }
  return absl::OkStatus();
}

ItemSelector::KeyWithProbability FifoSelector::Sample() {
  REVERB_CHECK(!keys_.empty());
  return {keys_.front(), 1.};
}

void FifoSelector::Clear() {
  keys_.clear();
  key_to_iterator_.clear();
}

KeyDistributionOptions FifoSelector::options() const {
  KeyDistributionOptions options;
  options.set_fifo(true);
  options.set_is_deterministic(true);
  return options;
}

std::string FifoSelector::DebugString() const { return "FifoSelector"; }

}  // namespace reverb
}  // namespace deepmind
