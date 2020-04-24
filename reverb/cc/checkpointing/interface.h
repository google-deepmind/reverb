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

#ifndef REVERB_CC_CHECKPOINTING_INTERFACE_H_
#define REVERB_CC_CHECKPOINTING_INTERFACE_H_

#include "reverb/cc/priority_table.h"

namespace deepmind {
namespace reverb {

// A checkpointer is able to encode the configuration, data and state as a
// proto . This proto is stored in a permanent storage system where it can
// retrieved at a later point and restore a copy of the checkpointed tables.
class CheckpointerInterface {
 public:
  virtual ~CheckpointerInterface() = default;

  // Save a new checkpoint for every table in `tables` to permanent storage. If
  // successful, `path` will contain an ABSOLUTE path that could be used to
  // restore the checkpoint.
  virtual tensorflow::Status Save(std::vector<PriorityTable*> tables,
                                  int keep_latest, std::string* path) = 0;

  // Attempts to load a checkpoint from the active workspace.
  //
  // Tables loaded from checkpoint must already exist in `tables`. When
  // constructing the newly loaded table the extensions are passed from the old
  // table and the item is replaced with the newly loaded table.
  virtual tensorflow::Status Load(
      absl::string_view relative_path, ChunkStore* chunk_store,
      std::vector<std::shared_ptr<PriorityTable>>* tables) = 0;

  // Finds the most recent checkpoint within the active workspace. See `Load`
  // for more details.
  virtual tensorflow::Status LoadLatest(
      ChunkStore* chunk_store,
      std::vector<std::shared_ptr<PriorityTable>>* tables) = 0;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_CHECKPOINTING_INTERFACE_H_
