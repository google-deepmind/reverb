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

#ifndef LEARNING_DEEPMIND_REPLAY_REVERB_SUPPORT_GRPC_UTIL_H_
#define LEARNING_DEEPMIND_REPLAY_REVERB_SUPPORT_GRPC_UTIL_H_

#include "grpcpp/grpcpp.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace deepmind {
namespace reverb {

constexpr char kStreamRemovedMessage[] = "Stream removed";

// Identify if the given grpc::Status corresponds to an HTTP stream removed
// error (see chttp2_transport.cc).
//
// When auto-reconnecting to a remote TensorFlow worker after it restarts, gRPC
// can return an UNKNOWN error code with a "Stream removed" error message.
// This should not be treated as an unrecoverable error.
//
// N.B. This is dependent on the error message from grpc remaining consistent.
inline bool IsStreamRemovedError(const ::grpc::Status& s) {
  return !s.ok() && s.error_code() == ::grpc::StatusCode::UNKNOWN &&
         s.error_message() == kStreamRemovedMessage;
}

inline grpc::Status ToGrpcStatus(const absl::Status& s) {
  if (s.ok()) return grpc::Status::OK;

  return grpc::Status(static_cast<grpc::StatusCode>(s.code()),
                      std::string(s.message()));
}

inline absl::Status FromGrpcStatus(const grpc::Status& s) {
  if (s.ok()) return absl::OkStatus();

  // Convert "UNKNOWN" stream removed errors into unavailable, to allow
  // for retry upstream.
  if (IsStreamRemovedError(s)) {
    return absl::UnavailableError(s.error_message());
  }
  return absl::Status(
      static_cast<absl::StatusCode>(s.error_code()), s.error_message());
}

inline std::string FormatGrpcStatus(const grpc::Status& s) {
  return absl::Substitute("[$0] $1", s.error_code(), s.error_message());
}

inline bool IsLocalhostOrInProcess(absl::string_view hostname) {
  return absl::StrContains(hostname, ":127.0.0.1:") ||
         absl::StrContains(hostname, "[::1]") || hostname == "unknown";
}

}  // namespace reverb
}  // namespace deepmind

#endif  // LEARNING_DEEPMIND_REPLAY_REVERB_SUPPORT_GRPC_UTIL_H_
