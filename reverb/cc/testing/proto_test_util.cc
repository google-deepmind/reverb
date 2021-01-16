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

#include "reverb/cc/testing/proto_test_util.h"

#include <vector>

#include "reverb/cc/platform/logging.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/tensor_compression.h"
#include "tensorflow/core/framework/tensor.h"

namespace deepmind {
namespace reverb {
namespace testing {

ChunkData MakeChunkData(uint64_t key) {
  return MakeChunkData(key, MakeSequenceRange(key * 100, 0, 1));
}

ChunkData MakeChunkData(uint64_t key, SequenceRange range) {
  ChunkData chunk;
  chunk.set_chunk_key(key);
  tensorflow::Tensor t(tensorflow::DT_INT32,
                       {range.end() - range.start() + 1, 10});
  t.flat<int32_t>().setConstant(1);
  CompressTensorAsProto(t, chunk.mutable_data()->add_tensors());
  *chunk.mutable_sequence_range() = std::move(range);

  return chunk;
}

SequenceRange MakeSequenceRange(uint64_t episode_id, int32_t start, int32_t end) {
  REVERB_CHECK_LE(start, end);
  SequenceRange sequence;
  sequence.set_episode_id(episode_id);
  sequence.set_start(start);
  sequence.set_end(end);
  return sequence;
}

KeyWithPriority MakeKeyWithPriority(uint64_t key, double priority) {
  KeyWithPriority update;
  update.set_key(key);
  update.set_priority(priority);
  return update;
}

PrioritizedItem MakePrioritizedItem(uint64_t key, double priority,
                                    const std::vector<ChunkData>& chunks) {
  QCHECK(!chunks.empty());

  PrioritizedItem item;
  item.set_key(key);
  item.set_priority(priority);

  for (const auto& chunk : chunks) {
    item.add_chunk_keys(chunk.chunk_key());
  }

  item.mutable_sequence_range()->set_length(
      1 + chunks.back().sequence_range().end() -
      chunks.front().sequence_range().start());

  return item;
}

}  // namespace testing
}  // namespace reverb
}  // namespace deepmind
