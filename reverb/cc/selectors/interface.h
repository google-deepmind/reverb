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

#ifndef REVERB_CC_SELECTORS_INTERFACE_H_
#define REVERB_CC_SELECTORS_INTERFACE_H_

#include <cstdint>

#include "absl/status/status.h"
#include "reverb/cc/checkpointing/checkpoint.pb.h"

namespace deepmind {
namespace reverb {

// Allows sampling from a population of keys with a specified priority per key.
//
// Member methods will not be called concurrently, so implementations do not
// need to be thread-safe.  More to the point, a number of subclasses use bit
// generators that are not thread-safe, so methods like `Sample` are not
// thread-safe.
class ItemSelector {
 public:
  using Key = uint64_t;

  struct KeyWithProbability {
    Key key;
    double probability;
  };

  virtual ~ItemSelector() = default;

  // Deletes a key and the associated priority. Returns an error if the key does
  // not exist.
  virtual absl::Status Delete(Key key) = 0;

  // Inserts a key and associated priority. Returns an error without any change
  // if the key already exists.
  virtual absl::Status Insert(Key key, double priority) = 0;

  // Updates a key and associated priority. Returns an error if the key does
  // not exist.
  virtual absl::Status Update(Key key, double priority) = 0;

  // Samples a key. Must contain keys when this is called.
  virtual KeyWithProbability Sample() = 0;

  // Clear the distribution of all data.
  virtual void Clear() = 0;

  // Options for dynamically constructing the distribution. Required when
  // reconstructing class from checkpoint.  Also used to query table metadata.
  virtual KeyDistributionOptions options() const = 0;

  // Returns a summary string description.
  virtual std::string DebugString() const = 0;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SELECTORS_INTERFACE_H_
