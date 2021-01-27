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

#ifndef REVERB_CC_PLATFORM_DEFAULT_STATUS_BUILDER_H_
#define REVERB_CC_PLATFORM_DEFAULT_STATUS_BUILDER_H_

#include <sstream>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "reverb/cc/platform/default/source_location.h"

namespace deepmind {
namespace reverb {
namespace internal {

class ABSL_MUST_USE_RESULT StatusBuilder {
 public:
  StatusBuilder(const StatusBuilder& sb);
  StatusBuilder& operator=(const StatusBuilder& sb);
  // Creates a `StatusBuilder` based on an original status.  If logging is
  // enabled, it will use `location` as the location from which the log message
  // occurs.  A typical user will call this with `DRISHTI_LOC`.
  StatusBuilder(const absl::Status& original_status,
                source_location location)
      : status_(original_status),
        line_(location.line()),
        file_(location.file_name()),
        stream_(new std::ostringstream) {}

  StatusBuilder(absl::Status&& original_status,
                source_location location)
      : status_(std::move(original_status)),
        line_(location.line()),
        file_(location.file_name()),
        stream_(new std::ostringstream) {}

  // Creates a `StatusBuilder` from a drishti status code.  If logging is
  // enabled, it will use `location` as the location from which the log message
  // occurs.  A typical user will call this with `DRISHTI_LOC`.
  StatusBuilder(absl::StatusCode code, source_location location)
      : status_(code, ""),
        line_(location.line()),
        file_(location.file_name()),
        stream_(new std::ostringstream) {}

  StatusBuilder(const absl::Status& original_status, const char* file, int line)
      : status_(original_status),
        line_(line),
        file_(file),
        stream_(new std::ostringstream) {}

  bool ok() const { return status_.ok(); }

  StatusBuilder& SetAppend();

  StatusBuilder& SetPrepend();

  StatusBuilder& SetNoLogging();

  template <typename T>
  StatusBuilder& operator<<(const T& msg) {
    if (status_.ok()) return *this;
    *stream_ << msg;
    return *this;
  }

  operator absl::Status() const&;
  operator absl::Status() &&;

  absl::Status JoinMessageToStatus();

 private:
  // Specifies how to join the error message in the original status and any
  // additional message that has been streamed into the builder.
  enum class MessageJoinStyle {
    kAnnotate,
    kAppend,
    kPrepend,
  };

  // The status that the result will be based on.
  absl::Status status_;
  // The line to record if this file is logged.
  int line_;
  // Not-owned: The file to record if this status is logged.
  const char* file_;
  bool no_logging_ = false;
  // The additional messages added with `<<`.
  std::unique_ptr<std::ostringstream> stream_;
  // Specifies how to join the message in `status_` and `stream_`.
  MessageJoinStyle join_style_ = MessageJoinStyle::kAnnotate;
};

inline StatusBuilder AlreadyExistsErrorBuilder(
    source_location location) {
  return StatusBuilder(absl::StatusCode::kAlreadyExists, location);
}

inline StatusBuilder FailedPreconditionErrorBuilder(
    source_location location) {
  return StatusBuilder(absl::StatusCode::kFailedPrecondition, location);
}

inline StatusBuilder InternalErrorBuilder(source_location location) {
  return StatusBuilder(absl::StatusCode::kInternal, location);
}

inline StatusBuilder InvalidArgumentErrorBuilder(
    source_location location) {
  return StatusBuilder(absl::StatusCode::kInvalidArgument, location);
}

inline StatusBuilder NotFoundErrorBuilder(source_location location) {
  return StatusBuilder(absl::StatusCode::kNotFound, location);
}

inline StatusBuilder UnavailableErrorBuilder(
    source_location location) {
  return StatusBuilder(absl::StatusCode::kUnavailable, location);
}

inline StatusBuilder UnimplementedErrorBuilder(
    source_location location) {
  return StatusBuilder(absl::StatusCode::kUnimplemented, location);
}

inline StatusBuilder UnknownErrorBuilder(source_location location) {
  return StatusBuilder(absl::StatusCode::kUnknown, location);
}

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_PLATFORM_DEFAULT_STATUS_BUILDER_H_
