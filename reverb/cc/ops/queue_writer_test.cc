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

#include "reverb/cc/ops/queue_writer.h"

#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_matchers.h"
#include "reverb/cc/support/signature.h"
#include "reverb/cc/trajectory_writer.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace deepmind {
namespace reverb {
namespace {

using Step = ::std::vector<::absl::optional<::tensorflow::Tensor>>;
using StepRef = ::std::vector<::absl::optional<::std::weak_ptr<CellRef>>>;

const auto kIntSpec = internal::TensorSpec{"0", tensorflow::DT_INT32, {1}};
const auto kFloatSpec = internal::TensorSpec{"0", tensorflow::DT_FLOAT, {1}};


inline std::string Int32Str() {
  return tensorflow::DataTypeString(tensorflow::DT_INT32);
}

inline tensorflow::Tensor MakeTensor(const internal::TensorSpec& spec) {
  if (spec.shape.dims() < 1) {
    return tensorflow::Tensor(spec.dtype, {});
  }

  tensorflow::TensorShape shape;
  REVERB_CHECK(spec.shape.AsTensorShape(&shape));
  tensorflow::Tensor tensor(spec.dtype, shape);

  for (int i = 0; i < tensor.NumElements(); i++) {
    if (spec.dtype == tensorflow::DT_FLOAT) {
      tensor.flat<float>()(i) = i;
    } else if (spec.dtype == tensorflow::DT_INT32) {
      tensor.flat<int32_t>()(i) = i;
    } else if (spec.dtype == tensorflow::DT_DOUBLE) {
      tensor.flat<double>()(i) = i;
    } else {
      REVERB_LOG(REVERB_FATAL) << "Unexpeted dtype";
    }
  }

  return tensor;
}

inline tensorflow::Tensor MakeRandomTensor(const internal::TensorSpec& spec) {
  auto tensor = MakeTensor(spec);

  absl::BitGen bit_gen;
  for (int i = 0; i < tensor.NumElements(); i++) {
    if (spec.dtype == tensorflow::DT_INT32) {
      tensor.flat<int32_t>()(i) = absl::Uniform<int32_t>(
          bit_gen, 0, std::numeric_limits<int32_t>::max());
    } else {
      REVERB_LOG(REVERB_FATAL) << "Unexpeted dtype";
    }
  }

  return tensor;
}

std::vector<TrajectoryColumn> MakeTrajectory(
    std::vector<std::vector<absl::optional<std::weak_ptr<CellRef>>>>
        trajectory) {
  std::vector<TrajectoryColumn> columns;
  for (const auto& optional_refs : trajectory) {
    std::vector<std::weak_ptr<CellRef>> col_refs;
    for (const auto& optional_ref : optional_refs) {
      col_refs.push_back(optional_ref.value());
    }
    columns.push_back(TrajectoryColumn(std::move(col_refs), /*squeeze=*/false));
  }
  return columns;
}

TEST(QueueWriter, AppendValidatesDtype) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/10, &queue);
  StepRef refs;

  // Initiate the spec with the first step.
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kFloatSpec)}), &refs));

  // Change the dtypes in the next step.
  auto status =
      writer.Append(Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}), &refs);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  absl::StrCat("Tensor of wrong dtype provided for column 1. "
                               "Got ",
                               Int32Str(), " but expected float.")));
}

TEST(QueueWriter, AppendValidatesShapes) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/10, &queue);
  StepRef refs;

  // Initiate the spec with the first step.
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &refs));

  // Change the dtypes in the next step.
  auto status = writer.Append(Step({MakeTensor(internal::TensorSpec{
                                  kIntSpec.name, kIntSpec.dtype, {3}})}),
                              &refs);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Tensor of incompatible shape provided for column 0. "
                  "Got [3] which is incompatible with [1]."));
}

TEST(QueueWriter, AppendAcceptsPartialSteps) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/10, &queue);

  // Initiate the spec with the first step.
  StepRef both;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kFloatSpec)}), &both));

  // Only append to the first column.
  StepRef first_column_only;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec), absl::nullopt}),
                                 &first_column_only));
  EXPECT_FALSE(first_column_only[1].has_value());
}

TEST(QueueWriter, AppendPartialRejectsMultipleUsesOfSameColumn) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/10, &queue);

  // Append first column only.
  StepRef first_column_only;
  REVERB_ASSERT_OK(
      writer.AppendPartial(Step({MakeTensor(kIntSpec)}), &first_column_only));

  // Appending the second column only should be fine.
  StepRef second_column_only;
  REVERB_ASSERT_OK(writer.AppendPartial(
      Step({absl::nullopt, MakeTensor(kFloatSpec)}), &second_column_only));

  // Appending the first column again should not be allowed.
  StepRef first_column_again;
  auto status =
      writer.AppendPartial(Step({MakeTensor(kIntSpec)}), &first_column_again);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Append/AppendPartial called with data containing column "
                  "that was present in previous AppendPartial call."));
}

TEST(QueueWriter,
     AppendRejectsColumnsProvidedInPreviousPartialCall) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/10, &queue);

  // Append first column only.
  StepRef first_column_only;
  REVERB_ASSERT_OK(
      writer.AppendPartial(Step({MakeTensor(kIntSpec)}), &first_column_only));

  // Finalize the step with BOTH columns. That is both the missing column and
  // the one that was already provided.
  StepRef both_columns;
  auto status = writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kFloatSpec)}), &both_columns);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Append/AppendPartial called with data containing column "
                  "that was present in previous AppendPartial call."));
}

TEST(QueueWriter, AppendPartialDoesNotIncrementEpisodeStep) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/10, &queue);
  // Append first column only and keep the step open.
  StepRef first_column_only;
  REVERB_ASSERT_OK(
      writer.AppendPartial(Step({MakeTensor(kIntSpec)}), &first_column_only));

  // Append to the second column only and close the step.
  StepRef second_column_only;
  REVERB_ASSERT_OK(writer.Append(Step({absl::nullopt, MakeTensor(kFloatSpec)}),
                                 &second_column_only));

  // Since the step was kept open after the first call the second call should
  // result in the same episode step.
  EXPECT_EQ(first_column_only[0]->lock()->episode_step(),
            second_column_only[1]->lock()->episode_step());

  // If we do another call then the episode step should have changed since the
  // previous step was closed with the second call.
  StepRef first_column_only_again;
  REVERB_ASSERT_OK(
      writer.Append(Step({MakeTensor(kIntSpec)}), &first_column_only_again));
  EXPECT_EQ(first_column_only[0]->lock()->episode_step() + 1,
            first_column_only_again[0]->lock()->episode_step());
}


TEST(QueueWriter, QueueIsEmptyIfNoItemsCreated) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/1, &queue);
  StepRef refs;

  for (int i = 0; i < 10; ++i) {
    REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &refs));
  }

  EXPECT_TRUE(queue.empty());
}

TEST(QueueWriter, ItemsAreAddedToQueue) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/1, &queue);
  StepRef refs;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &refs));

  EXPECT_TRUE(queue.empty());

  // The chunk is completed so inserting an item should result in both chunk
  // and item being sent.
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{refs[0]}})));
  EXPECT_EQ(queue.size(), 1);

  // Adding a second item should result in the item being sent straight away.
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 0.5, MakeTrajectory({{refs[0]}})));
  EXPECT_EQ(queue.size(), 2);

  // In the second step we only write to the first column.
  StepRef second;
  REVERB_ASSERT_OK(
      writer.Append(Step({MakeTensor(kIntSpec), absl::nullopt}), &second));



  for (auto trajectory : queue){
    EXPECT_EQ(trajectory.size(), 1);
    // Trajectories have an extra batch dimension
    EXPECT_EQ(trajectory[0].shape(), tensorflow::TensorShape({1, 1}));
    EXPECT_EQ(trajectory[0].dtype(), tensorflow::DT_INT32);
  }
}

TEST(QueueWriter, ItemsWithComplexTensorsWork) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/1, &queue);
  StepRef refs;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(internal::TensorSpec{
                                     kIntSpec.name, kIntSpec.dtype, {3}})}),
                                 &refs));
  EXPECT_TRUE(queue.empty());

  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{refs[0]}})));
  EXPECT_EQ(queue.size(), 1);

  for (auto trajectory : queue){
    EXPECT_EQ(trajectory.size(), 1);
    // Trajectories have an extra batch dimension
    EXPECT_EQ(trajectory[0].shape(), tensorflow::TensorShape({1, 3}));
    EXPECT_EQ(trajectory[0].dtype(), tensorflow::DT_INT32);
  }
}

TEST(QueueWriter, ItemsWithTwoStepsWork) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/2, &queue);
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(internal::TensorSpec{
                                     kIntSpec.name, kIntSpec.dtype, {3}})}),
                                 &first));
    StepRef second;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(internal::TensorSpec{
                                     kIntSpec.name, kIntSpec.dtype, {3}})}),
                                 &second));
  EXPECT_TRUE(queue.empty());

  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[0], second[0]}})));
  EXPECT_EQ(queue.size(), 1);

  for (auto trajectory : queue){
    EXPECT_EQ(trajectory.size(), 1);
    // The two tensors are batched
    EXPECT_EQ(trajectory[0].shape(), tensorflow::TensorShape({2, 3}));
    EXPECT_EQ(trajectory[0].dtype(), tensorflow::DT_INT32);
  }
}

TEST(QueueWriter, ItemCreatedEvenIfChunksAreNotComplete) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/2, &queue);

  // Write to both columns in the first step.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}), &first));

  // Create an item which references the first row in the two columns.
  REVERB_ASSERT_OK(writer.CreateItem("table", 1.0,
                                     MakeTrajectory({{first[0]}, {first[1]}})));
  EXPECT_EQ(queue.size(), 1);

  for (auto trajectory : queue){
    EXPECT_EQ(trajectory.size(), 2);
    for (auto column : trajectory){
      // Trajectories have an extra batch dimension
      EXPECT_EQ(column.shape(), tensorflow::TensorShape({1, 1}));
      EXPECT_EQ(column.dtype(), tensorflow::DT_INT32);
    }
  }
}


TEST(QueueWriter, CreateItemForcesOnlyReferencedChunks) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/2, &queue);


  // Write to both columns in the first step.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kIntSpec)}), &first));

  // Create an item which references the first row in second column.
  REVERB_ASSERT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{first[1]}})));

  // The trajectory is created even if the chunks are not completed.
  // In the chunker we use, chunkas are never really created.
  EXPECT_EQ(queue.size(), 1);

  EXPECT_FALSE(first[0].value().lock()->IsReady());
  EXPECT_FALSE(first[1].value().lock()->IsReady());
}


TEST(QueueWriter, CreateItemRejectsExpiredCellRefs) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/2, &queue);

  // Take three steps.
  StepRef first;
  StepRef second;
  StepRef third;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &second));
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &third));

  // The num_keep_alive_refs is set to 2 so the first step has expired.
  auto status = writer.CreateItem("table", 1.0, MakeTrajectory({{first[0]}}));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  "Error in column 0: Column contains expired CellRef."));
  // The second step is still available.
  REVERB_EXPECT_OK(
      writer.CreateItem("table", 1.0, MakeTrajectory({{second[0]}})));
}


TEST(QueueWriter, CreateItemValidatesTrajectoryDtype) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/2, &queue);

  // Take a step with two columns with different dtypes.
  StepRef step;
  REVERB_ASSERT_OK(writer.Append(
      Step({MakeTensor(kIntSpec), MakeTensor(kFloatSpec)}), &step));

  // Create a trajectory where the two dtypes are used in the same column.
  auto status =
      writer.CreateItem("table", 1.0, MakeTrajectory({{step[0], step[1]}}));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr(
                  absl::StrCat("Error in column 0: Column references tensors "
                               "with different dtypes: ",
                               Int32Str(), " (index 0) != float (index 1).")));
}

TEST(QueueWriter, CreateItemValidatesTrajectoryShapes) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/2, &queue);

  // Take a step with two columns with different shapes.
  StepRef step;

  REVERB_ASSERT_OK(writer.Append(
      Step({
          MakeTensor(kIntSpec),
          MakeTensor(internal::TensorSpec{"1", kIntSpec.dtype, {2}}),
      }),
      &step));

  // Create a trajectory where the two shapes are used in the same column.
  auto status =
      writer.CreateItem("table", 1.0, MakeTrajectory({{step[0], step[1]}}));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr("Error in column 0: Column references tensors with "
                           "incompatible shapes: [1] "
                           "(index 0) not compatible with [2] (index 1)."));
}

TEST(QueueWriter, CreateItemValidatesTrajectoryNotEmpty) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/1, &queue);


  StepRef step;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));

  // Create a trajectory without any columns.
  auto no_columns_status = writer.CreateItem("table", 1.0, {});
  EXPECT_EQ(no_columns_status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(no_columns_status.message()),
              ::testing::HasSubstr("trajectory must not be empty."));

  // Create a trajectory where all columns are empty.
  auto all_columns_empty_status = writer.CreateItem("table", 1.0, {{}, {}});
  EXPECT_EQ(all_columns_empty_status.code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(all_columns_empty_status.message()),
              ::testing::HasSubstr("trajectory must not be empty."));
}

TEST(QueueWriter, CreateItemValidatesSqueezedColumns) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/1, &queue);


  StepRef step;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));

  // Create a trajectory with a column that has two rows and is squeezed.
  auto status = writer.CreateItem(
      "table", 1.0,
      {TrajectoryColumn({step[0].value(), step[0].value()}, true)});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      std::string(status.message()),
      ::testing::HasSubstr("Error in column 0: TrajectoryColumn must contain "
                           "exactly one row when squeeze is set but got 2."));
}


TEST(QueueWriter, CreateItemWithSqueezedColumn) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/1, &queue);


  StepRef step;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));

  // Create a trajectory with a column that has rows and is squeezed.
  auto status = writer.CreateItem(
      "table", 1.0,
      {TrajectoryColumn({step[0].value()}, true)});
  REVERB_EXPECT_OK(status);
  EXPECT_EQ(queue.size(), 1);

  for (auto trajectory : queue){
    EXPECT_EQ(trajectory.size(), 1);
    for (auto column : trajectory){
      // Squeezed Columns don't have an extra batch dimension
      EXPECT_EQ(column.shape(), tensorflow::TensorShape({1}));
      EXPECT_EQ(column.dtype(), tensorflow::DT_INT32);
    }
  }
}

TEST(QueueWriter, EndEpisodeCanClearBuffers) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/2, &queue);


  // Take a step.
  StepRef step;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));

  // If we don't clear the buffer then the reference should be alive after.
  REVERB_ASSERT_OK(writer.EndEpisode(/*clear_buffers=*/false));
  EXPECT_FALSE(step[0]->expired());

  // If we clear the buffer then the reference should expire.
  REVERB_ASSERT_OK(writer.EndEpisode(/*clear_buffers=*/true));
  EXPECT_TRUE(step[0]->expired());
}

TEST(QueueWriter,
     EndEpisodeNeverFinalizesChunks) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/2, &queue);

  // Take a step.
  StepRef step;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));

  // The chunk is not yet finalized as `max_chunk_length` is 2.
  EXPECT_FALSE(step[0]->lock()->IsReady());

  // Calling `EndEpisode` should trigger the finalization of the chunk even if
  // it is not used by any item. Note that this is different from Flush which
  // only finalizes chunks which owns `CellRef`s that are referenced by pending
  // items.
  REVERB_ASSERT_OK(writer.EndEpisode(/*clear_buffers=*/false));
  EXPECT_FALSE(step[0]->lock()->IsReady());
}

TEST(QueueWriter, EndEpisodeResetsEpisodeKeyAndStep) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/2, &queue);


  // Take two steps in two different episodes.
  StepRef first;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &first));

  REVERB_ASSERT_OK(writer.EndEpisode(/*clear_buffers=*/false));

  StepRef second;
  REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &second));

  // Verify that the `episode_key` was changed between episodes and that the
  // episode step was reset to 0.
  EXPECT_NE(first[0]->lock()->episode_id(), second[0]->lock()->episode_id());
  EXPECT_EQ(first[0]->lock()->episode_step(), 0);
  EXPECT_EQ(second[0]->lock()->episode_step(), 0);
}

TEST(QueueWriter, EpisodeStepIsIncrementedByAppend) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/2, &queue);


  // The counter should start at 0.
  EXPECT_EQ(writer.episode_steps(), 0);

  // Each append call should increment the counter.
  for (int i = 1; i < 11; i++) {
    StepRef step;
    REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));
    EXPECT_EQ(writer.episode_steps(), i);
  }

  // AppendPartial should not have any impact on the counter.
  {
    StepRef step;
    REVERB_ASSERT_OK(writer.AppendPartial(Step({MakeTensor(kIntSpec)}), &step));
  }
  EXPECT_EQ(writer.episode_steps(), 10);

  // Closing the partial step should increment it though.
  {
    StepRef step;
    REVERB_ASSERT_OK(writer.Append(Step({}), &step));
  }
  EXPECT_EQ(writer.episode_steps(), 11);
}

TEST(QueueWriter, EpisodeStepIsNotIncrementedByAppendPartial) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/2, &queue);


  // The counter should start at 0.
  EXPECT_EQ(writer.episode_steps(), 0);
  // AppendPartial should not have any impact on the counter.
  {
    StepRef step;
    REVERB_ASSERT_OK(writer.AppendPartial(Step({MakeTensor(kIntSpec)}), &step));
  }
  EXPECT_EQ(writer.episode_steps(), 0);

  // Closing the partial step should increment it though.
  {
    StepRef step;
    REVERB_ASSERT_OK(writer.Append(Step({}), &step));
  }
  EXPECT_EQ(writer.episode_steps(), 1);
}

TEST(QueueWriter, EpisodeStepIsResetByEndEpisode) {
  std::deque<std::vector<tensorflow::Tensor>> queue;
  QueueWriter writer(/*num_keep_alive_refs=*/2, &queue);

  // Take a step and check that the counter has been incremented.
  {
    StepRef step;
    REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));
  }
  EXPECT_EQ(writer.episode_steps(), 1);

  // Ending the episode while clearing the buffers should reset the counter.
  REVERB_ASSERT_OK(writer.EndEpisode(true, absl::Milliseconds(100)));
  EXPECT_EQ(writer.episode_steps(), 0);

  // Repeat the process but don't clear the buffers when ending the episode.
  {
    StepRef step;
    REVERB_ASSERT_OK(writer.Append(Step({MakeTensor(kIntSpec)}), &step));
  }
  EXPECT_EQ(writer.episode_steps(), 1);
  REVERB_ASSERT_OK(writer.EndEpisode(false, absl::Milliseconds(100)));
  EXPECT_EQ(writer.episode_steps(), 0);
}


}  // namespace
}  // namespace reverb
}  // namespace deepmind
