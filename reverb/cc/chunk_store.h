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

#ifndef REVERB_CC_CHUNK_STORE_H_
#define REVERB_CC_CHUNK_STORE_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/schema.pb.h"

namespace deepmind {
namespace reverb {

// Maintains a bijection from chunk keys to Chunks. For inserting, the caller
// passes ChunkData which contains a chunk key and the actual data. We use the
// key for the mapping and wrap the ChunkData with in a thin class which
// provides a read-only accessor to the ChunkData.
//
//          +-----------+
//          |           |     shared_ptr
//          | ChunkData |<---------------------+
//          |           |                      |
//          +-----------+                      |
//                ^                            |
//                | shared_ptr                 |
//                |                            |
//         +--------------+              +----------+            +----------+
//         |              |   weak_ptr   |          | shared_ptr |          |
//         |  ChunkStore  |------------->|  Chunk   |<-----------|  Caller  |
//         |              |              |          |            |          |
//         +--------------+              +----------+            +----------+
//
// Data insertion is handled as follows:
//
//   1. Provided ChunkData is moved into a shared_ptr.
//   2. The data ptr is saved in an internal map and used to construct a
//      Chunk (heap allocated as a shared_ptr).
//   3. A weak_ptr is constructed from the shared_ptr of the Chunk and saved
//      in an internal map.
//   4. The shared_ptr of the Chunk is returned to the caller.
//
// Each Chunk is reference counted individually. When its reference count drops
// to zero, the Chunk is destroyed and subsequent calls to Get() will no longer
// return that Chunk. Please note that ChunkStore only holds a weak pointer to a
// Chunk, and thus does not count towards the reference count. For this reason,
// Insert() returns a shared pointer, as otherwise the Chunk would be destroyed
// right away.
//
// All public methods are thread safe.
class ChunkStore {
 public:
  using Key = uint64_t;

  class Chunk {
   public:
    explicit Chunk(ChunkData data);

    // Unique identifier of the chunk.
    uint64_t key() const;

    // Returns the proto data of the chunk.
    const ChunkData& data() const;

    // (Potentially cached) size of `data`.
    size_t DataByteSizeLong() const;

    // Size (bytes) of the tensors before compression. Alias for
    // `data().data_uncompressed_size()`.
    size_t uncompressed_data_size() const;

    // Alias for `data().sequence_range().episode_id()`.
    uint64_t episode_id() const;

    // The number of tensors batched together in each column. Note that all
    // columns always share the same number of rows (i.e batch dimension).
    int32_t num_rows() const;

    // Number of tensors in each step.
    int num_columns() const;

   private:
    ChunkData data_;
    mutable size_t data_byte_size_;
    mutable absl::once_flag data_byte_size_once_;
  };

  // Attempts to insert a Chunk into the map using the key inside `item`. If no
  // entry existed for the key, a new Chunk is created, inserted and returned.
  // Otherwise, the existing chunk is returned.
  std::shared_ptr<Chunk> Insert(ChunkData item) ABSL_LOCKS_EXCLUDED(mu_);

  // Gets the Chunk for each given key. Returns an error if one of the items
  // does not exist or if `Close` has been called. On success, the returned
  // items are in the same order as given in `keys`.
  absl::Status Get(absl::Span<const Key> keys,
                   std::vector<std::shared_ptr<Chunk>>* chunks)
      ABSL_LOCKS_EXCLUDED(mu_);

 private:
  // Gets an item. Returns nullptr if the item does not exist.
  std::shared_ptr<Chunk> GetItem(Key key) ABSL_SHARED_LOCKS_REQUIRED(mu_);

  // Holds the actual mapping of key to Chunk. We only hold a weak pointer to
  // the Chunk, which means that destruction and reference counting of the
  // chunks happens independently of this map.
  internal::flat_hash_map<Key, std::weak_ptr<Chunk>> data_ ABSL_GUARDED_BY(mu_);

  // Mutex protecting access to `data_`.
  mutable absl::Mutex mu_;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_CHUNK_STORE_H_
