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

#ifndef REVERB_CC_CLIENT_H_
#define REVERB_CC_CLIENT_H_

#include <stddef.h>

#include <memory>
#include <string>
#include <vector>

#include <cstdint>
#include "absl/base/thread_annotations.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/streaming_trajectory_writer.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/trajectory_writer.h"
#include "reverb/cc/writer.h"

namespace deepmind {
namespace reverb {

class Writer;

// See ReverbService proto definition for documentation.
class Client {
 public:
  struct ServerInfo {
    // This struct mirrors the ServerInfo message in
    // reverb_service.proto.  Take a look at that proto file for
    // field documentation.
    absl::uint128 tables_state_id;
    std::vector<TableInfo> table_info;
  };

  explicit Client(std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub);
  explicit Client(absl::string_view server_address);

  // Upon successful return, `writer` will contain an instance of Writer.
  absl::Status NewWriter(int chunk_length, int max_timesteps,
                         bool delta_encoded, std::unique_ptr<Writer>* writer);
  absl::Status NewWriter(int chunk_length, int max_timesteps,
                         bool delta_encoded,
                         absl::optional<int> max_in_flight_items,
                         std::unique_ptr<Writer>* writer);

  // Upon successful return, `sampler` will contain an instance of
  // Sampler.
  //
  // This version tries to look up `dtypes_and_shapes` for `table` via
  // `ServerInfo` call, and passes this to `sampler`.  The `validation_timeout`
  // parameter controls how long the wait is on `ServerInfo` if this data is
  // not already cached.
  absl::Status NewSampler(const std::string& table,
                          const Sampler::Options& options,
                          absl::Duration validation_timeout,
                          std::unique_ptr<Sampler>* sampler);

  // Upon successful return, `sampler` will contain an instance of
  // Sampler which does not perform any validation of shapes and dtypes.
  absl::Status NewSamplerWithoutSignatureCheck(
      const std::string& table, const Sampler::Options& options,
      std::unique_ptr<Sampler>* sampler);

  // Upon successful return, `sampler` will contain an instance of
  // Sampler.
  //
  // If the table has signature metadata available on the server, then
  // `validation_shapes` and `validation_dtypes` are checked against the
  // flattened signature.
  //
  // On the other hand, if the table info returned from the server lacks a
  // signature, then no validation is performed.  If no table entry exists
  // for the given table string, then a warning is logged.
  //
  // **NOTE** Because the sampler always prepends the entry key and
  // priority tensors when returning samples, the `validation_{dtypes, shapes}`
  // vectors must always be prepended with the signatures of these outputs.
  // Specifically, the user must pass as prefix the SampleInfo shapes and
  // dtypes.  See `GetDtypesAndShapesForSampler` for the most up-to-date
  // expected prefix.
  //
  // The remaining elements should be the dtypes/shapes of the entries
  // expected in table signature.
  //
  absl::Status NewSampler(
      const std::string& table, const Sampler::Options& options,
      const tensorflow::DataTypeVector& validation_dtypes,
      const std::vector<tensorflow::PartialTensorShape>& validation_shapes,
      absl::Duration validation_timeout, std::unique_ptr<Sampler>* sampler);

  // Simultaneously mutates priorities and deletes elements from replay table
  // `table`. If `timeout` is specified, function may return a
  // DEADLINE_EXCEEDED error. If `timeout` is not specified, function may block
  // indefinitely.
  absl::Status MutatePriorities(
      absl::string_view table, const std::vector<KeyWithPriority>& updates,
      const std::vector<uint64_t>& deletes,
      absl::Duration timeout = absl::InfiniteDuration());

  absl::Status Reset(const std::string& table);

  absl::Status Checkpoint(std::string* path);

  // Requests ServerInfo. Forces an update of internal signature caches.
  absl::Status ServerInfo(absl::Duration timeout, struct ServerInfo* info);
  // Waits indefinitely for server to respond.
  absl::Status ServerInfo(struct ServerInfo* info);

  // Validates `options` and if valid, creates a new `TrajectoryWriter`.
  //
  // TODO(b/177308010): Remove banner when `TrajectoryWriter` is ready for use.
  absl::Status NewTrajectoryWriter(const TrajectoryWriter::Options& options,
                                   std::unique_ptr<TrajectoryWriter>* writer);

  // This version tries to lookup and populate `options.flat_signature_map`
  // using a (potentially cached) `ServerInfo` call. `get_signature_timeout`
  // parameter controls how long the wait is on `ServerInfo` if this data is
  // not already cached.
  absl::Status NewTrajectoryWriter(const TrajectoryWriter::Options& options,
                                   absl::Duration get_signature_timeout,
                                   std::unique_ptr<TrajectoryWriter>* writer);

  // Validates `options` and if valid, creates a new
  // `StreamingTrajectoryWriter`.
  absl::Status NewStreamingTrajectoryWriter(
      const TrajectoryWriter::Options& options,
      std::unique_ptr<StreamingTrajectoryWriter>* writer);

 private:
  const std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub_;

  // Request direct access to Table managed by server. Result will only be
  // populated when the stub was created using a localhost address of a server
  // running in the same process.
  absl::Status GetLocalTablePtr(absl::string_view table_name,
                                std::shared_ptr<Table>* out);

  // Upon successful return, `sampler` will contain an instance of
  // Sampler.  This version is called by the public `NewSampler` methods.
  //
  // Care should be made to ensure that `dtypes_and_shapes` here includes
  // the prefix tensors associated with the SampleInfo.
  absl::Status NewSampler(const std::string& table,
                          const Sampler::Options& options,
                          internal::DtypesAndShapes dtypes_and_shapes,
                          std::unique_ptr<Sampler>* sampler);

  absl::Status MaybeUpdateServerInfoCache(
      absl::Duration timeout,
      std::shared_ptr<internal::FlatSignatureMap>* cached_flat_signatures);

  // Uses MaybeUpdateServerInfoCache to get ServerInfo and pull the
  // dtypes_and_shapes for `table`.  If `table` is not in the ServerInfo, then
  // dtypes_and_shapes is set to absl::nullopt.
  absl::Status GetDtypesAndShapesForSampler(
      const std::string& table, absl::Duration validation_timeout,
      internal::DtypesAndShapes* dtypes_and_shapes);

  // Purely functional request for server info.  Does not update any internal
  // caches.
  absl::Status GetServerInfo(absl::Duration timeout, struct ServerInfo* info);

  // Updates tables_state_id_ and cached_flat_signatures_ using info.
  absl::Status LockedUpdateServerInfoCache(const struct ServerInfo& info)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(cached_table_mu_);

  absl::Mutex cached_table_mu_;
  absl::uint128 tables_state_id_ ABSL_GUARDED_BY(cached_table_mu_);
  std::shared_ptr<internal::FlatSignatureMap> cached_flat_signatures_
      ABSL_GUARDED_BY(cached_table_mu_);
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_CLIENT_H_
