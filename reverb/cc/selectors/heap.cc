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

#include "reverb/cc/selectors/heap.h"

#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/selectors/interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {

HeapSelector::HeapSelector(bool min_heap)
    : sign_(min_heap ? 1 : -1), update_count_(0) {}

tensorflow::Status HeapSelector::Delete(ItemSelectorInterface::Key key) {
  auto it = nodes_.find(key);
  if (it == nodes_.end()) {
    return tensorflow::errors::InvalidArgument("Key ", key, " not found.");
  }
  heap_.Remove(it->second.get());
  nodes_.erase(it);
  return tensorflow::Status::OK();
}

tensorflow::Status HeapSelector::Insert(ItemSelectorInterface::Key key,
                                        double priority) {
  if (nodes_.contains(key)) {
    return tensorflow::errors::InvalidArgument("Key ", key,
                                               " already inserted.");
  }
  nodes_[key] =
      absl::make_unique<HeapNode>(key, priority * sign_, update_count_++);
  heap_.Push(nodes_[key].get());
  return tensorflow::Status::OK();
}

tensorflow::Status HeapSelector::Update(ItemSelectorInterface::Key key,
                                        double priority) {
  if (!nodes_.contains(key)) {
    return tensorflow::errors::InvalidArgument("Key ", key, " not found.");
  }
  nodes_[key]->priority = priority * sign_;
  nodes_[key]->update_number = update_count_++;
  heap_.Adjust(nodes_[key].get());
  return tensorflow::Status::OK();
}

ItemSelectorInterface::KeyWithProbability HeapSelector::Sample() {
  REVERB_CHECK(!nodes_.empty());
  return {heap_.top()->key, 1.};
}

void HeapSelector::Clear() {
  nodes_.clear();
  heap_.Clear();
}


KeyDistributionOptions HeapSelector::options() const {
  KeyDistributionOptions options;
  options.mutable_heap()->set_min_heap(sign_ == 1);
  options.set_is_deterministic(true);
  return options;
}

}  // namespace reverb
}  // namespace deepmind
