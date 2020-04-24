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

#include "reverb/cc/distributions/lifo.h"

#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/schema.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {

tensorflow::Status LifoDistribution::Delete(KeyDistributionInterface::Key key) {
  auto it = key_to_iterator_.find(key);
  if (it == key_to_iterator_.end())
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("Key ", key, " not found in distribution."));
  keys_.erase(it->second);
  key_to_iterator_.erase(it);
  return tensorflow::Status::OK();
}

tensorflow::Status LifoDistribution::Insert(KeyDistributionInterface::Key key,
                                            double priority) {
  if (key_to_iterator_.find(key) != key_to_iterator_.end()) {
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("Key ", key, " already exists in distribution."));
  }
  key_to_iterator_.emplace(key, keys_.emplace(keys_.begin(), key));
  return tensorflow::Status::OK();
}

tensorflow::Status LifoDistribution::Update(KeyDistributionInterface::Key key,
                                            double priority) {
  if (key_to_iterator_.find(key) == key_to_iterator_.end()) {
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("Key ", key, " not found in distribution."));
  }
  return tensorflow::Status::OK();
}

KeyDistributionInterface::KeyWithProbability LifoDistribution::Sample() {
  REVERB_CHECK(!keys_.empty());
  return {keys_.front(), 1.};
}

void LifoDistribution::Clear() {
  keys_.clear();
  key_to_iterator_.clear();
}

KeyDistributionOptions LifoDistribution::options() const {
  KeyDistributionOptions options;
  options.set_lifo(true);
  return options;
}

}  // namespace reverb
}  // namespace deepmind
