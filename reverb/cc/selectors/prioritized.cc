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

#include "reverb/cc/selectors/prioritized.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/interface.h"

namespace deepmind {
namespace reverb {
namespace {

// If the approximation error at a node exceeds this threshold, the sum tree is
// reinitialized.
constexpr double kMaxApproximationError = 1e-4;

// A priority of zero should correspond to zero probability, even if the
// priority exponent is zero. So this modified version of std::pow is used to
// turn priorities into weights. Expects base and exponent to be non-negative.
inline double power(double base, double exponent) {
  return base == 0. ? 0. : std::pow(base, exponent);
}

absl::Status CheckValidPriority(double priority) {
  if (std::isnan(priority))
    return absl::InvalidArgumentError("Priority must not be NaN.");
  if (priority < 0)
    return absl::InvalidArgumentError(
        "Priority must not be negative.");
  return absl::OkStatus();
}

}  // namespace

PrioritizedSelector::PrioritizedSelector(double priority_exponent,
                                         uint64_t seed)
    : priority_exponent_(priority_exponent),
      capacity_(std::pow(2, 17)),
      sum_tree_(capacity_),
      rng_(seed) {}

absl::Status PrioritizedSelector::Delete(Key key) {
  const size_t last_index = key_to_index_.size() - 1;
  const auto it = key_to_index_.find(key);
  if (it == key_to_index_.end())
    return absl::InvalidArgumentError(absl::StrCat("Key ", key, " not found."));
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

  return absl::OkStatus();
}

absl::Status PrioritizedSelector::Insert(Key key, double priority) {
  REVERB_RETURN_IF_ERROR(CheckValidPriority(priority));
  const size_t index = key_to_index_.size();
  if (index == capacity_) {
    capacity_ *= 2;
    sum_tree_.resize(capacity_);
  }
  if (!key_to_index_.try_emplace(key, index).second) {
    return absl::InvalidArgumentError(
        absl::StrCat("Key ", key, " already inserted."));
  }
  sum_tree_[index].key = key;

  SetNode(index, power(priority, priority_exponent_));
  return absl::OkStatus();
}

absl::Status PrioritizedSelector::Update(Key key, double priority) {
  REVERB_RETURN_IF_ERROR(CheckValidPriority(priority));
  const auto it = key_to_index_.find(key);
  if (it == key_to_index_.end()) {
    return absl::InvalidArgumentError(absl::StrCat("Key ", key, " not found."));
  }
  SetNode(it->second, power(priority, priority_exponent_));
  return absl::OkStatus();
}

ItemSelector::KeyWithProbability PrioritizedSelector::Sample() {
  const size_t size = key_to_index_.size();
  REVERB_CHECK_NE(size, 0);

  // This should never be called concurrently from multiple threads.
  const double target = uniform_distr_(rng_);  // [0.0, 1.0)
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
  REVERB_LOG_IF(REVERB_ERROR, target_weight >= picked_weight)
      << "Target weight should be smaller than picked weight (target_weight: "
      << target_weight << " >= picked_weight:" << picked_weight << ").";
  return {sum_tree_[index].key, picked_weight / total_weight};
}

void PrioritizedSelector::Clear() {
  for (size_t i = 0; i < key_to_index_.size(); ++i) {
    sum_tree_[i].sum = 0;
    sum_tree_[i].value = 0;
  }
  key_to_index_.clear();
}

KeyDistributionOptions PrioritizedSelector::options() const {
  KeyDistributionOptions options;
  options.mutable_prioritized()->set_priority_exponent(priority_exponent_);
  options.set_is_deterministic(false);
  return options;
}

std::string PrioritizedSelector::DebugString() const {
  return absl::StrCat(
      "PrioritizedSelector(priority_exponent=", priority_exponent_, ")");
}

double PrioritizedSelector::NodeValue(size_t index) const {
  return sum_tree_[index].value;
}

double PrioritizedSelector::NodeSum(size_t index) const {
  return index < key_to_index_.size() ? sum_tree_[index].sum : 0;
}

double PrioritizedSelector::NodeSumTestingOnly(size_t index) const {
  return NodeSum(index);
}

// Updates the sum stored in a node and tracks if the tree needs to be
// re-initialized.
// Ensure the sum never becomes negative (it may happen because of rounding
// errors).
#define UPDATE_SUM(i)                                          \
  sum_tree_[(i)].sum += difference;                            \
  if (sum_tree_[(i)].sum < 0) sum_tree_[(i)].sum = 0.0;        \
  error = std::abs(sum_tree_[(i)].sum - NodeSum(2 * (i) + 1) - \
                   NodeSum(2 * (i) + 2) - sum_tree_[(i)].value);

void PrioritizedSelector::SetNode(size_t index, double value) {
  const double difference = value - NodeValue(index);

  // The floating point approximation error of the last node update.
  double error = 0.0;

  // Update the subject node.
  sum_tree_[index].value = value;
  UPDATE_SUM(index);

  // Update all parents until we find the root node.
  while (index != 0 && error <= kMaxApproximationError) {
    index = (index - 1) / 2;
    UPDATE_SUM(index);
  }

  // If floating-point errors have built up, re-initialize the tree.
  if (error > kMaxApproximationError) {
    REVERB_LOG(REVERB_WARNING)
        << "Tree needs to be initialized because node with index " << index
        << " has approximation error " << error
        << ", which exceeds the threshold of " << kMaxApproximationError;
    ReinitializeSumTree();
  }
}

#undef UPDATE_SUM

void PrioritizedSelector::ReinitializeSumTree() {
  // Re-initialize the sums from the leaves to the root node.
  for (int64_t i = sum_tree_.size() - 1; i >= 0; --i) {
    sum_tree_[i].sum = NodeValue(i) + NodeSum(2 * i + 1) + NodeSum(2 * i + 2);
  }
}

}  // namespace reverb
}  // namespace deepmind
