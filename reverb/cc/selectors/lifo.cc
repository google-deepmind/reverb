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

#include "reverb/cc/selectors/lifo.h"

#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {

tensorflow::Status LifoSelector::Delete(ItemSelectorInterface::Key key) {
  auto it = key_to_iterator_.find(key);
  if (it == key_to_iterator_.end())
    return tensorflow::errors::InvalidArgument("Key ", key, " not found.");
  keys_.erase(it->second);
  key_to_iterator_.erase(it);
  return tensorflow::Status::OK();
}

tensorflow::Status LifoSelector::Insert(ItemSelectorInterface::Key key,
                                        double priority) {
  if (key_to_iterator_.find(key) != key_to_iterator_.end()) {
    return tensorflow::errors::InvalidArgument("Key ", key,
                                               " already inserted.");
  }
  key_to_iterator_.emplace(key, keys_.emplace(keys_.begin(), key));
  return tensorflow::Status::OK();
}

tensorflow::Status LifoSelector::Update(ItemSelectorInterface::Key key,
                                        double priority) {
  if (key_to_iterator_.find(key) == key_to_iterator_.end()) {
    return tensorflow::errors::InvalidArgument("Key ", key, " not found.");
  }
  return tensorflow::Status::OK();
}

ItemSelectorInterface::KeyWithProbability LifoSelector::Sample() {
  REVERB_CHECK(!keys_.empty());
  return {keys_.front(), 1.};
}

void LifoSelector::Clear() {
  keys_.clear();
  key_to_iterator_.clear();
}

KeyDistributionOptions LifoSelector::options() const {
  KeyDistributionOptions options;
  options.set_lifo(true);
  return options;
}

}  // namespace reverb
}  // namespace deepmind
