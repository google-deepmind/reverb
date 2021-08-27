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

#include "reverb/cc/support/state_statistics.h"

#include "gtest/gtest.h"

namespace deepmind {
namespace reverb {
namespace internal {
namespace {

enum class States {
  kState1,
  kState2,
  kState3,
};

TEST(StateStatisticsTest, StateStatistics) {
  StateStatistics<States> stats;
  EXPECT_EQ(absl::Seconds(0), stats.GetTotalTimeIn(States::kState1));
  EXPECT_EQ(absl::Seconds(0), stats.GetTotalTimeIn(States::kState2));
  EXPECT_EQ(absl::Seconds(0), stats.GetTotalTimeIn(States::kState3));
  stats.Enter(States::kState1);
  EXPECT_EQ(stats.CurrentState(), States::kState1);
  EXPECT_LT(absl::Seconds(0), stats.GetTotalTimeIn(States::kState1));
  EXPECT_EQ(absl::Seconds(0), stats.GetTotalTimeIn(States::kState2));
  EXPECT_EQ(absl::Seconds(0), stats.GetTotalTimeIn(States::kState3));
  stats.Enter(States::kState2);
  EXPECT_EQ(stats.CurrentState(), States::kState2);
  auto state1_time = stats.GetTotalTimeIn(States::kState1);
  auto state2_time = stats.GetTotalTimeIn(States::kState2);
  EXPECT_LT(absl::Seconds(0), state1_time);
  EXPECT_LT(absl::Seconds(0), state2_time);
  EXPECT_EQ(absl::Seconds(0), stats.GetTotalTimeIn(States::kState3));
  stats.Enter(States::kState3);
  EXPECT_EQ(stats.CurrentState(), States::kState3);
  EXPECT_EQ(state1_time, stats.GetTotalTimeIn(States::kState1));
  EXPECT_LT(state2_time, stats.GetTotalTimeIn(States::kState2));
  EXPECT_LT(absl::Seconds(0), stats.GetTotalTimeIn(States::kState3));
}


}  // namespace
}  // namespace internal
}  // namespace reverb
}  // namespace deepmind
