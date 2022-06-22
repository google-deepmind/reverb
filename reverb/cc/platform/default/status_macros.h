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

#ifndef REVERB_CC_PLATFORM_DEFAULT_STATUS_H_
#define REVERB_CC_PLATFORM_DEFAULT_STATUS_H_

// IWYU pragma: private
#include "absl/status/status.h"
#include "reverb/cc/platform/logging.h"

// Evaluates an expression that produces a `absl::Status`. If the status
// is not ok, returns it from the current function.
//
// For example:
//   absl::Status MultiStepFunction() {
//     REVERB_RETURN_IF_ERROR(Function(args...));
//     REVERB_RETURN_IF_ERROR(foo.Method(args...));
//     return absl::OkStatus();
//   }
//
//
// Another example:
//   absl::Status MultiStepFunction() {
//     REVERB_RETURN_IF_ERROR(Function(args...)) << "in MultiStepFunction";
//     REVERB_RETURN_IF_ERROR(foo.Method(args...)).Log(base_logging::ERROR)
//         << "while processing query: " << query.DebugString();
//     return absl::OkStatus();
//   }
//
#define REVERB_RETURN_IF_ERROR(...)                        \
  do {                                                     \
    ::absl::Status _status = (__VA_ARGS__);                \
    if (ABSL_PREDICT_FALSE(!_status.ok())) return _status; \
  } while (0)

#define REVERB_STATUS_MACROS_CONCAT_NAME(x, y) \
  REVERB_STATUS_MACROS_CONCAT_IMPL(x, y)
#define REVERB_STATUS_MACROS_CONCAT_IMPL(x, y) x##y

// Executes an expression `rexpr` that returns a `absl::StatusOr<T>`. On
// OK, extracts its value into the variable defined by `lhs`, otherwise returns
// from the current function. By default the error status is returned
// unchanged, but it may be modified by an `error_expression`. If there is an
// error, `lhs` is not evaluated; thus any side effects that `lhs` may have
// only occur in the success case.
//
// Interface:
//
//   REVERB_ASSIGN_OR_RETURN(lhs, rexpr)
//   REVERB_ASSIGN_OR_RETURN(lhs, rexpr, error_expression);
//
// WARNING: expands into multiple statements; it cannot be used in a single
// statement (e.g. as the body of an if statement without {})!
//
// Example: Declaring and initializing a new variable (ValueType can be anything
//          that can be initialized with assignment, including references):
//   REVERB_ASSIGN_OR_RETURN(ValueType value, MaybeGetValue(arg));
//
// Example: Assigning to an existing variable:
//   ValueType value;
//   REVERB_ASSIGN_OR_RETURN(value, MaybeGetValue(arg));
//
// Example: Assigning to an expression with side effects:
//   MyProto data;
//   REVERB_ASSIGN_OR_RETURN(*data.mutable_str(), MaybeGetValue(arg));
//   // No field "str" is added on error.
//
// Example: Assigning to a std::unique_ptr.
//   REVERB_ASSIGN_OR_RETURN(std::unique_ptr<T> ptr, MaybeGetPtr(arg));
//
// If passed, the `error_expression` is evaluated to produce the return
// value. The expression may reference any variable visible in scope.
// The expression may, however, evaluate to any type
// returnable by the function, including (void).
//
#define REVERB_ASSIGN_OR_RETURN(lhs, rexpr)                                 \
  REVERB_ASSIGN_OR_RETURN_IMPL(                                             \
      REVERB_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, \
      rexpr)

#define REVERB_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                                 \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) {                \
    return statusor.status();                              \
  }                                                        \
  lhs = std::move(statusor.value())

#define REVERB_CHECK_OK(val) REVERB_CHECK_EQ(::absl::OkStatus(), (val))

#endif  // REVERB_CC_PLATFORM_DEFAULT_STATUS_H_
