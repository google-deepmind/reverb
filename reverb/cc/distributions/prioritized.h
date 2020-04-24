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

#ifndef LEARNING_DEEPMIND_REPLAY_REVERB_DISTRIBUTIONS_PRIORITIZED_H_
#define LEARNING_DEEPMIND_REPLAY_REVERB_DISTRIBUTIONS_PRIORITIZED_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/distributions/interface.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {

// This an implementation of a categorical distribution that allows incremental
// changes to the keys to be made efficiently. The probability of sampling a key
// is proportional to its priority raised to configurable exponent.
//
// Since the priorities and probabilities are stored as doubles, numerical
// rounding errors may be introduced especially when the relative size of
// probabilities for keys is large. Ideally when using this class priorities are
// roughly the same scale and the priority exponent is not large, e.g. less than
// 2.
//
// This was forked from:
// ## proportional_picker.h
//
class PrioritizedDistribution : public KeyDistributionInterface {
 public:
  explicit PrioritizedDistribution(double priority_exponent);

  // O(log n) time.
  tensorflow::Status Delete(Key key) override;

  // The priority must be non-negative. O(log n) time.
  tensorflow::Status Insert(Key key, double priority) override;

  // The priority must be non-negative. O(log n) time.
  tensorflow::Status Update(Key key, double priority) override;

  // O(log n) time.
  KeyWithProbability Sample() override;

  // O(n) time.
  void Clear() override;

  KeyDistributionOptions options() const override;

 private:
  struct Node {
    Key key;
    // Sum of the exponentiated priority of this node and all its descendants.
    // This includes the entire sub tree with inner and leaf nodes.
    // `NodeValue()` can be used to get the exponentiated priority of a node
    // without its children.
    double sum = 0;
  };

  // Gets the individual value of a node in `sum_tree_` without the summed up
  // value of all its descendants.
  double NodeValue(size_t index) const;

  // Sum of the exponentiated priority of this node and all its descendants.
  // If the index is out of bounds, then 0 is returned.
  double NodeSum(size_t index) const;

  // Sets the individual value of a node in the `sum_tree_`. This does not
  // include the value of the descendants.
  void SetNode(size_t index, double value);

  // Controls the degree of prioritization. Priorities are raised to this
  // exponent before adding them to the `SumTree` as weights. A non-negative
  // number where a value of zero corresponds each key having the same
  // probability (except for keys with zero priority).
  const double priority_exponent_;

  // Capacity of the summary tree. Starts at ~130000 and grows exponentially.
  size_t capacity_;

  // A tree stored as a flat vector were each node is the sum of its children
  // plus its own exponentiated priority.
  std::vector<Node> sum_tree_;

  // Maps a key to the index where this key can be found in `sum_tree_`.
  absl::flat_hash_map<Key, size_t> key_to_index_;

  // Used for sampling, not thread-safe.
  absl::BitGen bit_gen_;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // LEARNING_DEEPMIND_REPLAY_REVERB_DISTRIBUTIONS_PRIORITIZED_H_
