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
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "reverb/cc/reverb_service.grpc.pb.h"
#include "reverb/cc/reverb_service.pb.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/schema.pb.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/writer.h"
#include "tensorflow/core/lib/core/status.h"

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
  tensorflow::Status NewWriter(int chunk_length, int max_timesteps,
                               bool delta_encoded,
                               std::unique_ptr<Writer>* writer);

  // Upon successful return, `sampler` will contain an instance of
  // Sampler.
  tensorflow::Status NewSampler(const std::string& table,
                                const Sampler::Options& options,
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
  // Specifically, the user must pass:
  //
  //   validation_dtypes[0:1] = {DT_UINT64, DT_DOUBLE}
  //   validation_shapes[0:1] = {PartialTensorShape({}), PartialTensorShape({})}
  //
  // and the remaining elements should be the dtypes/shapes of the entries
  // expected in table signature.
  //
  tensorflow::Status NewSampler(
      const std::string& table, const Sampler::Options& options,
      const tensorflow::DataTypeVector& validation_dtypes,
      const std::vector<tensorflow::PartialTensorShape>& validation_shapes,
      absl::Duration validation_timeout, std::unique_ptr<Sampler>* sampler);

  tensorflow::Status MutatePriorities(
      absl::string_view table, const std::vector<KeyWithPriority>& updates,
      const std::vector<uint64_t>& deletes);

  tensorflow::Status Reset(const std::string& table);

  tensorflow::Status Checkpoint(std::string* path);

  // Requests ServerInfo. Forces an update of internal signature caches.
  tensorflow::Status ServerInfo(absl::Duration timeout,
                                struct ServerInfo* info);
  // Waits indefinetely for server to respond.
  tensorflow::Status ServerInfo(struct ServerInfo* info);

 private:
  const std::shared_ptr</* grpc_gen:: */ReverbService::StubInterface> stub_;

  tensorflow::Status MaybeUpdateServerInfoCache(
      absl::Duration timeout,
      std::shared_ptr<internal::FlatSignatureMap>* cached_flat_signatures);

  // Purely functional request for server info.  Does not update any internal
  // caches.
  tensorflow::Status GetServerInfo(absl::Duration timeout,
                                   struct ServerInfo* info);

  // Updates tables_state_id_ and cached_flat_signatures_ using info.
  tensorflow::Status LockedUpdateServerInfoCache(const struct ServerInfo& info)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(cached_table_mu_);

  absl::Mutex cached_table_mu_;
  absl::uint128 tables_state_id_ ABSL_GUARDED_BY(cached_table_mu_);
  std::shared_ptr<internal::FlatSignatureMap> cached_flat_signatures_
      ABSL_GUARDED_BY(cached_table_mu_);
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_CLIENT_H_
