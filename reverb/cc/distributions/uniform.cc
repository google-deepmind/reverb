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

#include "reverb/cc/distributions/uniform.h"

#include "absl/strings/str_cat.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/schema.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {

tensorflow::Status UniformDistribution::Delete(Key key) {
  const auto it = key_to_index_.find(key);
  if (it == key_to_index_.end())
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("Key ", key, " not found in distribution."));
  const size_t index = it->second;
  key_to_index_.erase(it);

  const size_t last_index = keys_.size() - 1;
  const Key last_key = keys_.back();
  if (index != last_index) {
    keys_[index] = last_key;
    key_to_index_[last_key] = index;
  }

  keys_.pop_back();
  return tensorflow::Status::OK();
}

tensorflow::Status UniformDistribution::Insert(Key key, double priority) {
  const size_t index = keys_.size();
  if (!key_to_index_.emplace(key, index).second)
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("Key ", key, " already exists in distribution."));
  keys_.push_back(key);
  return tensorflow::Status::OK();
}

tensorflow::Status UniformDistribution::Update(Key key, double priority) {
  if (key_to_index_.find(key) == key_to_index_.end())
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("Key ", key, " not found in distribution."));
  return tensorflow::Status::OK();
}

KeyDistributionInterface::KeyWithProbability UniformDistribution::Sample() {
  REVERB_CHECK(!keys_.empty());

  // This code is not thread-safe, because bit_gen_ is not protected by a mutex
  // and is not itself thread-safe.
  const size_t index = absl::Uniform<size_t>(bit_gen_, 0, keys_.size());
  return {keys_[index], 1.0 / static_cast<double>(keys_.size())};
}

void UniformDistribution::Clear() {
  keys_.clear();
  key_to_index_.clear();
}

KeyDistributionOptions UniformDistribution::options() const {
  KeyDistributionOptions options;
  options.set_uniform(true);
  return options;
}

}  // namespace reverb
}  // namespace deepmind
