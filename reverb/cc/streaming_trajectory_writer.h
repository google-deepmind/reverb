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

#ifndef REVERB_CC_STREAMING_TRAJECTORY_WRITER_H_
#define REVERB_CC_STREAMING_TRAJECTORY_WRITER_H_

#include <stdint.h>

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/hash_set.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/key_generators.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/trajectory_writer.h"
#include "tensorflow/core/framework/tensor.h"

namespace deepmind::reverb {

// A streaming trajectory writer behaves similarly to the "standard" trajectory
// writer. It allows creating replay items of arbitrary length (i.e. the
// trajectory length doesn't need to be specified in advance). Many
// (overlapping) items can be created per episode.
//
// The major difference between `StreamingTrajectoryWriter` and
// `TrajectoryWriter` is how and when data is sent to the replay buffer. This
// class sends chunks as soon as they have reached a specified length (called
// chunk size) whereas `TrajectoryWriter` only sends chunks when they are
// referenced by a priority item. Especially when doing full-episode replay,
// trajectories can become very long, which leads to memory issues on the
// writing side if the entire trajectory is kept in memory until the priority
// item is created. In these situations, `StreamingTrajectoryWriter` can reduce
// memory consumption on the writing side. Note, however, that using long
// episodes may still pose problems on the reading side where the entire
// trajectory must be kept in memory.
//
// The downside of using `StreamingTrajectoryWriter` is that it does not
// gracefully recover from data loss. If streaming a chunk belonging to an
// episode fails, the write call will not be retried. Instead, the entire
// episode is considered corrupt and no priority items can be created until a
// new episode is started. If an episode is corrupt, calls for adding additional
// tensors or creating priority items will return an `absl::DataLossError`.
// Callers should start a new episode when they observe the error.
//
// This class is not thread-safe.
//
// TODO(b/143277674): Consolidate this implementation with TrajectoryWriter.
class StreamingTrajectoryWriter : public ColumnWriter {
 public:
  // Multiple `ChunkData` can be sent with the same `InsertStreamRequest`. If
  // the size of the message exceeds this value then the request is sent and the
  // remaining chunks are sent with other messages.
  static constexpr int64_t kMaxRequestSizeBytes = 40 * 1024 * 1024;  // 40MB.

  StreamingTrajectoryWriter(
      std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub,
      const TrajectoryWriter::Options& options);

  // Flushes pending items and then closes stream. If `Close` has already been
  // called then no action is taken.
  ~StreamingTrajectoryWriter() override;

  // See `ColumnWriter::Append` in trajectory_writer.h.
  absl::Status Append(
      std::vector<absl::optional<tensorflow::Tensor>> data,
      std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs) override;

  // See `ColumnWriter::AppendPartial` in trajectory_writer.h.
  absl::Status AppendPartial(
      std::vector<absl::optional<tensorflow::Tensor>> data,
      std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs);

  // See `ColumnWriter::CreateItem` in trajectory_writer.h.
  absl::Status CreateItem(
      absl::string_view table, double priority,
      absl::Span<const TrajectoryColumn> trajectory) override;

  // Terminates the current episode by clearing any recoverable errors and
  // resetting the episode state (i.e generates a new episode ID and sets step
  // index to 0).
  // TODO(b/143277674): Arguments are currently ignored.
  absl::Status EndEpisode(
      bool clear_buffers,
      absl::Duration timeout = absl::InfiniteDuration()) override;

  absl::Status EndEpisode() { return EndEpisode(true); }

 private:
  using InsertStream = grpc::ClientReaderWriterInterface<InsertStreamRequest,
                                                         InsertStreamResponse>;

  // See `Append` and `AppendPartial`.
  absl::Status AppendInternal(
      std::vector<absl::optional<tensorflow::Tensor>> data,
      bool increment_episode_step,
      std::vector<absl::optional<std::weak_ptr<CellRef>>>* refs);

  // Streams all chunks associated with the given references. Only streams
  // chunks that are ready.
  absl::Status StreamChunks(const std::vector<std::shared_ptr<CellRef>>& refs);

  // Sets `context_` and opens a gRPC InsertStream to the server iff the writer
  // has not yet been closed.
  void SetContextAndCreateStream();

  // Build and write the item insertion request to the stream. All chunks
  // referenced by item must have been written to the stream before calling
  // this method.
  absl::Status SendItem(PrioritizedItem item);

  // Writes the requests provided in the argument to the stream. If writing
  // fails, the current episode is considered corrupt, which is a recoverable
  // error, which will be reset when a new episode is started.
  absl::Status WriteStream(const InsertStreamRequest& request);

  // Stub used to create InsertStream gRPC streams.
  std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub_;

  // Configuration options.
  TrajectoryWriter::Options options_;

  // Used to generates keys for episode and item IDs.
  internal::UniformKeyGenerator key_generator_;

  // Mapping from column index to Chunker. Shared pointers are used as the
  // `CellRef`s created by the chunker will own a weak_ptr created using
  // `weak_from_this()` on the Chunker.
  internal::flat_hash_map<int, std::shared_ptr<Chunker>> chunkers_;

  // Set of chunk IDs belonging to this episode that have been streamed to the
  // replay buffer.
  internal::flat_hash_set<uint64_t> streamed_chunk_keys_;

  // ID of the active episode.
  uint64_t episode_id_;

  // Step within the episode.
  int episode_step_;

  // Set if a non transient error encountered by the stream worker or if `Close`
  // has been called. In the latter case `unrecoverable_status_` will be set to
  // `CancelledError`.
  absl::Status unrecoverable_error_;

  // Set if a a transient error occurs. Since this writer doesn't have a retry
  // mechanism, all data associated with the episode in which the transient
  // error occurred is considered corrupt. When a new episode starts, the error
  // is reset.
  absl::Status recoverable_error_;

  // Context used to create (and cancel) the gRPC stream used in
  // `stream_worker_`. The worker creates the context before invoking
  // `RunStreamWorker`. The mutex protects against potential data races between
  // concurrent `Close` calls and creation of new streams.
  std::unique_ptr<grpc::ClientContext> context_;

  // Active gRPC stream instance. Created on class construction and recreated if
  // a transient error occurrs (e.g. writing a request fails).
  std::unique_ptr<InsertStream> stream_;
};

}  // namespace deepmind::reverb

#endif  // REVERB_CC_STREAMING_TRAJECTORY_WRITER_H_
