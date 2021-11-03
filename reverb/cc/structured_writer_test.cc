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

#include "reverb/cc/structured_writer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "reverb/cc/patterns.pb.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/testing/proto_test_util.h"

namespace deepmind::reverb {
namespace {

// TODO(b/204562045): Test that patterns are applied correctly.
// TODO(b/204562045): Test that conditions are respected.

inline StructuredWriterConfig MakeConfig(const std::string& text_proto) {
  return testing::ParseTextProtoOrDie<StructuredWriterConfig>(text_proto);
}

MATCHER_P2(StatusIs, code, message, "") {
  return arg.code() == code && absl::StrContains(arg.message(), message);
}

TEST(ValidateStructuredWriterConfig, Valid_NoStart) {
  REVERB_EXPECT_OK(ValidateStructuredWriterConfig(MakeConfig(
      R"pb(
        flat { flat_source_index: 0 stop: -1 }
        table: "table"
        priority: 1.0
        conditions { buffer_length: true ge: 1 }
      )pb")));
}

TEST(ValidateStructuredWriterConfig, Valid_WithStartAndStop) {
  REVERB_EXPECT_OK(ValidateStructuredWriterConfig(MakeConfig(
      R"pb(
        flat { flat_source_index: 0 start: -2 stop: -1 }
        table: "table"
        priority: 1.0
        conditions { buffer_length: true ge: 2 }
      )pb")));
}

TEST(ValidateStructuredWriterConfig, Valid_WithStartAndNoStop) {
  REVERB_EXPECT_OK(ValidateStructuredWriterConfig(MakeConfig(
      R"pb(
        flat { flat_source_index: 0 start: -2 }
        table: "table"
        priority: 1.0
        conditions { buffer_length: true ge: 2 }
      )pb")));
}

TEST(ValidateStructuredWriterConfig, NoStartAndNoStop) {
  EXPECT_THAT(
      ValidateStructuredWriterConfig(MakeConfig(
          R"pb(
            flat { flat_source_index: 0 }
            table: "table"
            priority: 1.0
          )pb")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "At least one of `start` and `stop` must be specified."));
}

TEST(ValidateStructuredWriterConfig, NegativeFlatSourceIndex) {
  EXPECT_THAT(ValidateStructuredWriterConfig(MakeConfig(
                  R"pb(
                    flat { flat_source_index: -1 }
                    table: "table"
                    priority: 1.0
                  )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`flat_source_index` must be >= 0 but got -1."));
}

TEST(ValidateStructuredWriterConfig, ZeroStart) {
  EXPECT_THAT(ValidateStructuredWriterConfig(MakeConfig(R"pb(
                flat { flat_source_index: 0 start: 0 }
                table: "table"
                priority: 1.0
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`start` must be < 0 but got 0."));
}

TEST(ValidateStructuredWriterConfig, PositiveStart) {
  EXPECT_THAT(ValidateStructuredWriterConfig(MakeConfig(R"pb(
                flat { flat_source_index: 0 start: 1 }
                table: "table"
                priority: 1.0
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`start` must be < 0 but got 1."));
}

TEST(ValidateStructuredWriterConfig, PositiveStop) {
  EXPECT_THAT(ValidateStructuredWriterConfig(MakeConfig(R"pb(
                flat { flat_source_index: 0 start: -1 stop: 1 }
                table: "table"
                priority: 1.0
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`stop` must be <= 0 but got 1."));
}

TEST(ValidateStructuredWriterConfig, StopEqualToStart) {
  EXPECT_THAT(
      ValidateStructuredWriterConfig(MakeConfig(R"pb(
        flat { flat_source_index: 0 start: -2 stop: -2 }
        table: "table"
        priority: 1.0
      )pb")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "`stop` (-2) must be > `start` (-2) when both are specified."));
}

TEST(ValidateStructuredWriterConfig, StopLessThanStart) {
  EXPECT_THAT(
      ValidateStructuredWriterConfig(MakeConfig(R"pb(
        flat { flat_source_index: 0 start: -2 stop: -3 }
        table: "table"
        priority: 1.0
      )pb")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "`stop` (-3) must be > `start` (-2) when both are specified."));
}

TEST(ValidateStructuredWriterConfig, ZeroStopAndNoStart) {
  EXPECT_THAT(ValidateStructuredWriterConfig(MakeConfig(R"pb(
                flat { flat_source_index: 0 stop: 0 }
                table: "table"
                priority: 1.0
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`stop` must be < 0 when `start` isn't set but got 0."));
}

TEST(ValidateStructuredWriterConfig, NoBufferLengthCondition) {
  EXPECT_THAT(
      ValidateStructuredWriterConfig(MakeConfig(R"pb(
        flat { flat_source_index: 0 stop: -1 }
        table: "table"
        priority: 1.0
      )pb")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Config does not contain required buffer length condition;"));
}

TEST(ValidateStructuredWriterConfig, TooSmallBufferLengthCondition_SingleNode) {
  EXPECT_THAT(
      ValidateStructuredWriterConfig(MakeConfig(R"pb(
        flat { flat_source_index: 0 stop: -2 }
        table: "table"
        priority: 1.0
        conditions { buffer_length: true ge: 1 }
      )pb")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Config does not contain required buffer length condition;"));
}

TEST(ValidateStructuredWriterConfig, TooSmallBufferLengthCondition_MultiNode) {
  EXPECT_THAT(
      ValidateStructuredWriterConfig(MakeConfig(R"pb(
        flat { flat_source_index: 0 stop: -2 }
        flat { flat_source_index: 0 start: -3 }
        table: "table"
        priority: 1.0
        conditions { buffer_length: true ge: 2 }
      )pb")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Config does not contain required buffer length condition;"));
}

TEST(ValidateStructuredWriterConfig, Valid_TooLargeBufferLength_SingleNode) {
  REVERB_EXPECT_OK(ValidateStructuredWriterConfig(MakeConfig(R"pb(
    flat { flat_source_index: 0 stop: -2 }
    table: "table"
    priority: 1.0
    conditions { buffer_length: true ge: 3 }
  )pb")));
}

TEST(ValidateStructuredWriterConfig, Valid_TooLargeBufferLength_MultiNode) {
  REVERB_EXPECT_OK(ValidateStructuredWriterConfig(MakeConfig(R"pb(
    flat { flat_source_index: 0 stop: -2 }
    flat { flat_source_index: 0 stop: -1 }
    table: "table"
    priority: 1.0
    conditions { buffer_length: true ge: 3 }
  )pb")));
}

TEST(ValidateStructuredWriterConfig, NoLeftInCondition) {
  EXPECT_THAT(ValidateStructuredWriterConfig(MakeConfig(R"pb(
                flat { flat_source_index: 0 stop: -2 }
                table: "table"
                priority: 1.0
                conditions { ge: 2 }
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Conditions must specify a value for `left`"));
}

TEST(ValidateStructuredWriterConfig, NegativeModuloInCondition) {
  EXPECT_THAT(ValidateStructuredWriterConfig(MakeConfig(R"pb(
                flat { flat_source_index: 0 stop: -2 }
                table: "table"
                priority: 1.0
                conditions {
                  step_index: true
                  mod_eq { mod: -2 eq: 0 }
                }
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`mod_eq.mod` must be > 0 but got -2."));
}

TEST(ValidateStructuredWriterConfig, ZeroModuloInCondition) {
  EXPECT_THAT(ValidateStructuredWriterConfig(MakeConfig(R"pb(
                flat { flat_source_index: 0 stop: -2 }
                table: "table"
                priority: 1.0
                conditions {
                  step_index: true
                  mod_eq { mod: 0 eq: 0 }
                }
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`mod_eq.mod` must be > 0 but got 0."));
}

TEST(ValidateStructuredWriterConfig, NegativeModuloEqInCondition) {
  EXPECT_THAT(ValidateStructuredWriterConfig(MakeConfig(R"pb(
                flat { flat_source_index: 0 stop: -2 }
                table: "table"
                priority: 1.0
                conditions {
                  step_index: true
                  mod_eq { mod: 2 eq: -1 }
                }
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`mod_eq.eq` must be >= 0 but got -1."));
}

TEST(ValidateStructuredWriterConfig, NoCmpInCondition) {
  EXPECT_THAT(ValidateStructuredWriterConfig(MakeConfig(R"pb(
                flat { flat_source_index: 0 stop: -2 }
                table: "table"
                priority: 1.0
                conditions { step_index: true }
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Conditions must specify a value for `cmp`."));
}

TEST(ValidateStructuredWriterConfig, Valid_EndOfEpisodeCondition) {
  REVERB_EXPECT_OK(ValidateStructuredWriterConfig(MakeConfig(R"pb(
    flat { flat_source_index: 0 stop: -2 }
    table: "table"
    priority: 1.0
    conditions { buffer_length: true ge: 2 }
    conditions { is_end_episode: true eq: 1 }
  )pb")));
}

TEST(ValidateStructuredWriterConfig, EndOfEpisode_NotUsingEqOne) {
  auto valid = MakeConfig(R"pb(
    flat { flat_source_index: 0 stop: -2 }
    table: "table"
    priority: 1.0
    conditions { buffer_length: true ge: 2 }
    conditions { is_end_episode: true eq: 1 }
  )pb");
  REVERB_ASSERT_OK(ValidateStructuredWriterConfig(valid));

  StructuredWriterConfig ge = valid;
  ge.mutable_conditions(1)->set_ge(1);
  EXPECT_THAT(
      ValidateStructuredWriterConfig(ge),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Condition must use `eq=1` when using `is_end_episode`"));

  StructuredWriterConfig eq_zero = valid;
  eq_zero.mutable_conditions(1)->set_eq(0);
  EXPECT_THAT(
      ValidateStructuredWriterConfig(eq_zero),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Condition must use `eq=1` when using `is_end_episode`"));

  StructuredWriterConfig eq_two = valid;
  eq_two.mutable_conditions(1)->set_eq(2);
  EXPECT_THAT(
      ValidateStructuredWriterConfig(eq_two),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Condition must use `eq=1` when using `is_end_episode`"));

  StructuredWriterConfig le = valid;
  le.mutable_conditions(1)->set_le(1);
  EXPECT_THAT(
      ValidateStructuredWriterConfig(le),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Condition must use `eq=1` when using `is_end_episode`"));
}

TEST(ValidateStructuredWriterConfig, FlatIsEmpty) {
  EXPECT_THAT(ValidateStructuredWriterConfig(MakeConfig(R"pb(
                table: "table"
                priority: 1.0
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`flat` must not be empty."));
}

TEST(ValidateStructuredWriterConfig, TableIsEmpty) {
  EXPECT_THAT(ValidateStructuredWriterConfig(MakeConfig(R"pb(
                flat { flat_source_index: 0 stop: -2 }
                priority: 1.0
                conditions { buffer_length: true ge: 2 }
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`table` must not be empty."));
}

TEST(ValidateStructuredWriterConfig, NegativePriority) {
  EXPECT_THAT(ValidateStructuredWriterConfig(MakeConfig(R"pb(
                flat { flat_source_index: 0 stop: -2 }
                table: "table"
                priority: -1.0
                conditions { buffer_length: true ge: 2 }
              )pb")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "`priority` must be >= 0 but got -1.0"));
}

}  // namespace
}  // namespace deepmind::reverb
