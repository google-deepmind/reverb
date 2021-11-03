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

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "reverb/cc/chunker.h"
#include "reverb/cc/patterns.pb.h"
#include "reverb/cc/platform/hash_map.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/platform/status_macros.h"
#include "reverb/cc/trajectory_writer.h"

namespace deepmind::reverb {
namespace {

inline bool operator==(const CellRef::EpisodeInfo& lhs,
                       const CellRef::EpisodeInfo& rhs) {
  return lhs.episode_id == rhs.episode_id && lhs.step == rhs.step;
}

inline bool operator!=(const CellRef::EpisodeInfo& lhs,
                       const CellRef::EpisodeInfo& rhs) {
  return !(lhs == rhs);
}

inline bool HasStart(const PatternNode& node) {
  return node.start_or_none_case() == PatternNode::kStart;
}

inline bool HasStop(const PatternNode& node) {
  return node.stop_or_none_case() == PatternNode::kStop;
}

inline int MaxAge(const PatternNode& node) {
  return std::abs(HasStart(node) ? node.start() : node.stop());
}

CellRef::EpisodeInfo GetEpisodeInfo(
    const std::vector<std::deque<std::shared_ptr<CellRef>>>& columns) {
  for (const auto& col : columns) {
    if (!col.empty() && col.back() != nullptr) {
      return {col.back()->episode_id(), col.back()->episode_step()};
    }
  }

  REVERB_CHECK(false)
      << "This should never happen. Please contact the Reverb team.";
}

bool CheckCondition(
    const std::vector<std::deque<std::shared_ptr<CellRef>>>& columns,
    const CellRef::EpisodeInfo& current_step, int steps_since_applied,
    bool is_end_of_episode, const Condition& condition) {
  REVERB_CHECK(!columns.empty()) << "This should never happen";

  int left = [&]() -> int {
    switch (condition.left_case()) {
      case Condition::kStepIndex:
        return current_step.step;
      case Condition::kStepsSinceApplied:
        return steps_since_applied;
      case Condition::kBufferLength:
        return std::max_element(columns.begin(), columns.end(),
                                [](const auto& a, const auto& b) {
                                  return a.size() < b.size();
                                })
            ->size();
      case Condition::kIsEndEpisode:
        return is_end_of_episode ? 1 : 0;

      case Condition::LEFT_NOT_SET:
        REVERB_CHECK(false) << "This should never happen";
    }
  }();

  switch (condition.cmp_case()) {
    case Condition::kEq:
      return left == condition.eq();
    case Condition::kGe:
      return left >= condition.ge();
    case Condition::kLe:
      return left <= condition.le();
    case Condition::kModEq:
      return left % condition.mod_eq().mod() == condition.mod_eq().eq();
    case Condition::CMP_NOT_SET:
      REVERB_CHECK(false) << "This should never happen";
  }
}

absl::StatusOr<std::vector<TrajectoryColumn>> BuildTrajectory(
    const StructuredWriterConfig& config,
    const std::vector<std::deque<std::shared_ptr<CellRef>>>& columns) {
  std::vector<TrajectoryColumn> out;
  out.reserve(columns.size());

  for (const auto& node : config.flat()) {
    REVERB_QCHECK_GT(columns.size(), node.flat_source_index());
    const auto& col = columns[node.flat_source_index()];

    const int offset =
        col.size() + (HasStart(node) ? node.start() : node.stop());
    const int length = HasStart(node) ? node.stop() - node.start() : 1;

    auto it = col.begin();
    std::advance(it, offset);

    std::vector<std::weak_ptr<CellRef>> refs(length);
    for (int i = 0; i < length; i++) {
      if (*it == nullptr) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "The %dth column contain null values in the references slice",
            node.flat_source_index()));
      }
      refs[i] = *it;
      it++;
    }

    out.emplace_back(std::move(refs), !HasStart(node) && HasStop(node));
  }

  return out;
}

absl::Status ValidateCondition(const Condition& condition) {
  if (condition.left_case() == Condition::LEFT_NOT_SET) {
    return absl::InvalidArgumentError(
        "Conditions must specify a value for `left`.");
  }

  if (condition.cmp_case() == Condition::kModEq) {
    const auto& mod_eq = condition.mod_eq();
    if (mod_eq.mod() <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "`mod_eq.mod` must be > 0 but got %d.", mod_eq.mod()));
    }
    if (mod_eq.eq() < 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("`mod_eq.eq` must be >= 0 but got %d.", mod_eq.eq()));
    }
  }

  if (condition.cmp_case() == Condition::CMP_NOT_SET) {
    return absl::InvalidArgumentError(
        "Conditions must specify a value for `cmp`.");
  }

  if (condition.is_end_episode() && condition.eq() != 1) {
    return absl::InvalidArgumentError(
        "Condition must use `eq=1` when using `is_end_episode`.");
  }

  return absl::OkStatus();
}

absl::Status ValidatePatternNode(const PatternNode& node) {
  if (node.flat_source_index() < 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("`flat_source_index` must be >= 0 but got %d.",
                        node.flat_source_index()));
  }
  if (!HasStart(node) && !HasStop(node)) {
    return absl::InvalidArgumentError(
        "At least one of `start` and `stop` must be specified.");
  }
  if (HasStart(node) && node.start() >= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("`start` must be < 0 but got %d.", node.start()));
  }
  if (HasStop(node) && node.stop() > 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("`stop` must be <= 0 but got %d.", node.stop()));
  }
  if (HasStart(node) && HasStop(node) && node.start() >= node.stop()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`stop` (%d) must be > `start` (%d) when both are specified.",
        node.stop(), node.start()));
  }
  if (HasStop(node) && node.stop() == 0 && !HasStart(node)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`stop` must be < 0 when `start` isn't set but got %d.", node.stop()));
  }

  return absl::OkStatus();
}

std::vector<int> MaxHistoryLengthPerColumn(
    const std::vector<StructuredWriterConfig>& configs) {
  std::vector<int> max_age;
  for (const auto& config : configs) {
    for (const auto& node : config.flat()) {
      const auto idx = node.flat_source_index();
      while (max_age.size() <= idx) {
        max_age.push_back(0);
      }

      max_age[idx] = std::max(max_age[idx], MaxAge(node));
    }
  }
  return max_age;
}

}  // namespace

absl::Status ValidateStructuredWriterConfig(
    const StructuredWriterConfig& config) {
  if (config.flat_size() == 0) {
    return absl::InvalidArgumentError("`flat` must not be empty.");
  }
  if (config.table().empty()) {
    return absl::InvalidArgumentError("`table` must not be empty.");
  }
  if (config.priority() < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`priority` must be >= 0 but got %f.", config.priority()));
  }

  for (const auto& node : config.flat()) {
    REVERB_RETURN_IF_ERROR(ValidatePatternNode(node));
  }
  if (config.table().empty()) {
    return absl::InvalidArgumentError("`table` must not be empty.");
  }
  for (const auto& condition : config.conditions()) {
    REVERB_RETURN_IF_ERROR(ValidateCondition(condition));
  }

  // In order to avoid segfault it is neccessary for every pattern to include a
  // condition that checks that the buffer contains enough steps to build the
  // speficied trajectory.
  const int max_age = MaxAge(*std::max_element(
      config.flat().begin(), config.flat().end(),
      [](const auto& a, const auto& b) { return MaxAge(a) < MaxAge(b); }));

  if (std::none_of(config.conditions().begin(), config.conditions().end(),
                   [&](const auto& cond) {
                     return cond.buffer_length() && cond.ge() >= max_age;
                   })) {
    Condition want;
    want.set_buffer_length(true);
    want.set_ge(max_age);
    return absl::InvalidArgumentError(absl::StrFormat(
        "Config does not contain required buffer length condition;\n"
        "Config: \n%s\ndoes not contain:\n%s",
        config.DebugString(), want.DebugString()));
  }

  return absl::OkStatus();
}

StructuredWriter::StructuredWriter(std::unique_ptr<ColumnWriter> writer,
                                   std::vector<StructuredWriterConfig> configs)
    : writer_(std::move(writer)),
      max_column_history_(MaxHistoryLengthPerColumn(configs)) {
  for (auto& config : configs) {
    REVERB_CHECK_OK(ValidateStructuredWriterConfig(config));
    configs_and_states_.push_back({std::move(config)});
  }
}

absl::Status StructuredWriter::Append(
    std::vector<absl::optional<tensorflow::Tensor>> data) {
  return AppendInternal(std::move(data), true);
}

absl::Status StructuredWriter::AppendPartial(
    std::vector<absl::optional<tensorflow::Tensor>> data) {
  return AppendInternal(std::move(data), false);
}

absl::Status StructuredWriter::AppendInternal(
    std::vector<absl::optional<tensorflow::Tensor>> data, bool finalize_step) {
  // Forward the data to the writer. Note that the writer is response for
  // checking that the same column is not populated multiple during the same
  // step.
  std::vector<absl::optional<std::weak_ptr<CellRef>>> refs;
  if (finalize_step) {
    REVERB_RETURN_IF_ERROR(writer_->Append(std::move(data), &refs));
  } else {
    REVERB_RETURN_IF_ERROR(writer_->AppendPartial(std::move(data), &refs));
  }

  for (int i = 0; i < refs.size(); i++) {
    // The remaining columns are not used by any of the patterns.
    if (i >= max_column_history_.size()) {
      break;
    }

    // Make sure we have a buffer for the column.
    while (i >= columns_.size()) {
      columns_.emplace_back();
    }

    // Append the reference, if any, to the column buffer.
    if (refs[i].has_value()) {
      columns_[i].push_back(refs[i]->lock());
    } else {
      columns_[i].push_back(nullptr);
    }

    // Free references to cells that never will be used.
    if (columns_[i].size() > max_column_history_[i]) {
      columns_[i].pop_front();
    }
  }

  return ApplyConfigs(/*is_end_of_episode=*/false);
}

absl::Status StructuredWriter::ApplyConfigs(bool is_end_of_episode) {
  if (columns_.empty() || columns_.front().empty()) {
    return absl::OkStatus();
  }

  CellRef::EpisodeInfo current_step = GetEpisodeInfo(columns_);

  for (auto& c : configs_and_states_) {
    // Never apply the same pattern twice on the same step.
    if (c.last_applied == current_step) {
      continue;
    }

    // Only increment `steps_since_applied` ONCE per step.
    if (c.last_checked != current_step) {
      c.steps_since_applied++;
      c.last_checked = current_step;
    }

    // Don't do anything unless all conditions are fulfilled.
    if (std::any_of(c.config.conditions().begin(), c.config.conditions().end(),
                    [&](const auto& cond) {
                      return !CheckCondition(columns_, current_step,
                                             c.steps_since_applied,
                                             is_end_of_episode, cond);
                    })) {
      continue;
    }

    // Try to build the trajectory. If it turns out that it contained null
    // values then we behave just as if a static condition wasn't fulfilled.
    auto trajectory_or_status = BuildTrajectory(c.config, columns_);
    if (absl::IsFailedPrecondition(trajectory_or_status.status())) {
      continue;
    }

    REVERB_ASSIGN_OR_RETURN(auto trajectory, trajectory_or_status);
    REVERB_RETURN_IF_ERROR(writer_->CreateItem(
        c.config.table(), c.config.priority(), std::move(trajectory)));

    c.last_applied = current_step;
    c.steps_since_applied = 0;
  }

  return absl::OkStatus();
}

absl::Status StructuredWriter::EndEpisode(bool clear_buffers,
                                          absl::Duration timeout) {
  REVERB_RETURN_IF_ERROR(ApplyConfigs(/*is_end_of_episode=*/true));
  REVERB_RETURN_IF_ERROR(writer_->EndEpisode(clear_buffers, timeout));
  if (clear_buffers) {
    columns_.clear();
  }

  return absl::OkStatus();
}

absl::Status StructuredWriter::Flush(int ignore_last_num_items,
                                     absl::Duration timeout) {
  return writer_->Flush(ignore_last_num_items, timeout);
}

}  // namespace deepmind::reverb
