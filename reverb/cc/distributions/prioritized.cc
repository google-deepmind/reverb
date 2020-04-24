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

#include "reverb/cc/distributions/prioritized.h"

#include <cmath>
#include <cstddef>

#include "absl/random/distributions.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/schema.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {
namespace {

// A priority of zero should correspond to zero probability, even if the
// priority exponent is zero. So this modified version of std::pow is used to
// turn priorities into weights. Expects base and exponent to be non-negative.
inline double power(double base, double exponent) {
  return base == 0. ? 0. : std::pow(base, exponent);
}

tensorflow::Status CheckValidPriority(double priority) {
  if (std::isnan(priority))
    return tensorflow::errors::InvalidArgument("Priority must not be NaN.");
  if (priority < 0)
    return tensorflow::errors::InvalidArgument(
        "Priority must not be negative.");
  return tensorflow::Status::OK();
}

}  // namespace

PrioritizedDistribution::PrioritizedDistribution(double priority_exponent)
    : priority_exponent_(priority_exponent), capacity_(std::pow(2, 17)) {
  REVERB_CHECK_GE(priority_exponent_, 0);
  sum_tree_.resize(capacity_);
}

tensorflow::Status PrioritizedDistribution::Delete(Key key) {
  const size_t last_index = key_to_index_.size() - 1;
  const auto it = key_to_index_.find(key);
  if (it == key_to_index_.end())
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("Key ", key, " not found in distribution."));
  const size_t index = it->second;

  if (index != last_index) {
    // Replace the element that we want to remove with the last element.
    SetNode(index, NodeValue(last_index));
    const Key last_key = sum_tree_[last_index].key;
    sum_tree_[index].key = last_key;
    key_to_index_[last_key] = index;
  }

  SetNode(last_index, 0);
  key_to_index_.erase(it);  // Note that this must occur after SetNode.

  return tensorflow::Status::OK();
}

tensorflow::Status PrioritizedDistribution::Insert(Key key, double priority) {
  TF_RETURN_IF_ERROR(CheckValidPriority(priority));
  const size_t index = key_to_index_.size();
  if (index == capacity_) {
    capacity_ *= 2;
    sum_tree_.resize(capacity_);
  }
  if (!key_to_index_.try_emplace(key, index).second) {
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("Key ", key, " already exists in distribution."));
  }
  sum_tree_[index].key = key;
  REVERB_CHECK_EQ(sum_tree_[index].sum, 0);
  SetNode(index, power(priority, priority_exponent_));
  return tensorflow::Status::OK();
}

tensorflow::Status PrioritizedDistribution::Update(Key key, double priority) {
  TF_RETURN_IF_ERROR(CheckValidPriority(priority));
  const auto it = key_to_index_.find(key);
  if (it == key_to_index_.end()) {
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("Key ", key, " not found in distribution."));
  }
  SetNode(it->second, power(priority, priority_exponent_));
  return tensorflow::Status::OK();
}

KeyDistributionInterface::KeyWithProbability PrioritizedDistribution::Sample() {
  const size_t size = key_to_index_.size();
  REVERB_CHECK_NE(size, 0);

  // This should never be called concurrently from multiple threads.
  const double target = absl::Uniform<double>(bit_gen_, 0, 1);
  const double total_weight = sum_tree_[0].sum;

  // All keys have zero priority so treat as if uniformly sampling.
  if (total_weight == 0) {
    const size_t pos = static_cast<size_t>(target * size);
    return {sum_tree_[pos].key, 1. / size};
  }

  // We begin traversing the `sum_tree_` from the root to the children in order
  // to find the `index` corresponding to the sampled `target_weight`.
  size_t index = 0;
  double target_weight = target * total_weight;
  while (true) {
    // Go to the left sub tree if it contains our sampled `target_weight`.
    const size_t left_index = 2 * index + 1;
    const double left_sum = NodeSum(left_index);
    if (target_weight < left_sum) {
      index = left_index;
      continue;
    }
    target_weight -= left_sum;
    // Go to the right sub tree if it contains our sampled `target_weight`.
    const size_t right_index = 2 * index + 2;
    const double right_sum = NodeSum(right_index);
    if (target_weight < right_sum) {
      index = right_index;
      continue;
    }
    target_weight -= right_sum;
    // Otherwise it is the current index.
    break;
  }
  REVERB_CHECK_LT(index, size);
  const double picked_weight = NodeValue(index);
  REVERB_CHECK_LT(target_weight, picked_weight);
  return {sum_tree_[index].key, picked_weight / total_weight};
}

void PrioritizedDistribution::Clear() {
  for (size_t i = 0; i < key_to_index_.size(); ++i) {
    sum_tree_[i].sum = 0;
  }
  key_to_index_.clear();
}

KeyDistributionOptions PrioritizedDistribution::options() const {
  KeyDistributionOptions options;
  options.mutable_prioritized()->set_priority_exponent(priority_exponent_);
  return options;
}

double PrioritizedDistribution::NodeValue(size_t index) const {
  const size_t left_index = 2 * index + 1;
  const size_t right_index = 2 * index + 2;
  return sum_tree_[index].sum - NodeSum(left_index) - NodeSum(right_index);
}

double PrioritizedDistribution::NodeSum(size_t index) const {
  return index < key_to_index_.size() ? sum_tree_[index].sum : 0;
}

void PrioritizedDistribution::SetNode(size_t index, double value) {
  double difference = value - NodeValue(index);
  sum_tree_[index].sum += difference;
  while (index != 0) {
    index = (index - 1) / 2;
    sum_tree_[index].sum += difference;
  }
}

}  // namespace reverb
}  // namespace deepmind
