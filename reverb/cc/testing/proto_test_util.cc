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
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace deepmind {
namespace reverb {
namespace testing {

ChunkData MakeChunkData(uint64_t key) {
  return MakeChunkData(key, MakeSequenceRange(key * 100, 0, 1), 1);
}

ChunkData MakeChunkData(uint64_t key, SequenceRange range) {
  return MakeChunkData(key, range, 1);
}

ChunkData MakeChunkData(uint64_t key, SequenceRange range, int num_tensors) {
  ChunkData chunk;
  chunk.set_chunk_key(key);
  tensorflow::Tensor t(tensorflow::DT_INT32,
                       {range.end() - range.start() + 1, 10});
  t.flat<int32_t>().setConstant(1);
  for (int i = 0; i < num_tensors; i++) {
    CompressTensorAsProto(t, chunk.mutable_data()->add_tensors());
  }
  *chunk.mutable_sequence_range() = std::move(range);

  return chunk;
}

SequenceRange MakeSequenceRange(uint64_t episode_id, int32_t start,
                                int32_t end) {
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

  for (int i = 0; i < chunks.front().data().tensors_size(); i++) {
    auto* col = item.mutable_flat_trajectory()->add_columns();
    for (const auto& chunk : chunks) {
      auto* slice = col->add_chunk_slices();
      slice->set_chunk_key(chunk.chunk_key());
      slice->set_offset(0);
      slice->set_length(chunk.data().tensors(0).tensor_shape().dim(0).size());
      slice->set_index(i);
    }
  }

  return item;
}

PrioritizedItem MakePrioritizedItem(const std::string& table, uint64_t key,
                                    double priority,
                                    const std::vector<ChunkData>& chunks) {
  auto item = MakePrioritizedItem(key, priority, chunks);
  item.set_table(table);
  return item;
}

}  // namespace testing
}  // namespace reverb
}  // namespace deepmind
