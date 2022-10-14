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

#include "reverb/cc/platform/checkpointing_utils.h"

#include <memory>

#include "reverb/cc/checkpointing/checkpoint.pb.h"
#include "reverb/cc/checkpointing/interface.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/selectors/heap.h"
#include "reverb/cc/selectors/interface.h"
#include "reverb/cc/selectors/lifo.h"
#include "reverb/cc/selectors/prioritized.h"
#include "reverb/cc/selectors/uniform.h"

namespace deepmind {
namespace reverb {

std::unique_ptr<ItemSelector> MakeSelector(
    const KeyDistributionOptions& options) {
  switch (options.distribution_case()) {
    case KeyDistributionOptions::kFifo:
      return std::make_unique<FifoSelector>();
    case KeyDistributionOptions::kLifo:
      return std::make_unique<LifoSelector>();
    case KeyDistributionOptions::kUniform:
      return std::make_unique<UniformSelector>();
    case KeyDistributionOptions::kPrioritized:
      return std::make_unique<PrioritizedSelector>(
          options.prioritized().priority_exponent());
    case KeyDistributionOptions::kHeap:
      return std::make_unique<HeapSelector>(options.heap().min_heap());
    case KeyDistributionOptions::DISTRIBUTION_NOT_SET:
      REVERB_LOG(REVERB_FATAL) << "Selector not set";
  }
}

}  // namespace reverb
}  // namespace deepmind
