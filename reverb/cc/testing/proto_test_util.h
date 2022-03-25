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

#ifndef REVERB_CC_TESTING_PROTO_TEST_UTIL_H_
#define REVERB_CC_TESTING_PROTO_TEST_UTIL_H_

#include <vector>

#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gmock/gmock.h"
#include "reverb/cc/platform/logging.h"
#include "reverb/cc/schema.pb.h"

namespace deepmind {
namespace reverb {
namespace testing {

ChunkData MakeChunkData(uint64_t key);
ChunkData MakeChunkData(uint64_t key, SequenceRange range);
ChunkData MakeChunkData(uint64_t key, SequenceRange range, int num_tensors);

SequenceRange MakeSequenceRange(uint64_t episode_id, int32_t start, int32_t end);

KeyWithPriority MakeKeyWithPriority(uint64_t key, double priority);

PrioritizedItem MakePrioritizedItem(uint64_t key, double priority,
                                    const std::vector<ChunkData>& chunks);

PrioritizedItem MakePrioritizedItem(const std::string& table, uint64_t key,
                                    double priority,
                                    const std::vector<ChunkData>& chunks);

// Simple implementation of a proto matcher comparing string representations.
//
// IMPORTANT: Only use this for protos whose textual representation is
// deterministic (that may not be the case for the map collection type).

class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const std::string& expected)
      : expected_proto_str_(expected) {}
  explicit ProtoStringMatcher(const google::protobuf::Message& expected)
      : expected_proto_str_(expected.DebugString()) {}

  template <typename Message>
  bool MatchAndExplain(const Message& actual_proto,
                       ::testing::MatchResultListener* listener) const;

  void DescribeTo(::std::ostream* os) const { *os << expected_proto_str_; }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: " << expected_proto_str_;
  }

  void SetComparePartially() {
    scope_ = ::google::protobuf::util::MessageDifferencer::PARTIAL;
  }

 private:
  const std::string expected_proto_str_;
  google::protobuf::util::MessageDifferencer::Scope scope_ =
      google::protobuf::util::MessageDifferencer::FULL;
};

template <typename T>
T CreateProto(const std::string& textual_proto) {
  T proto;
  REVERB_CHECK(google::protobuf::TextFormat::ParseFromString(textual_proto, &proto));
  return proto;
}

template <typename Message>
bool ProtoStringMatcher::MatchAndExplain(
    const Message& actual_proto,
    ::testing::MatchResultListener* listener) const {
  Message expected_proto = CreateProto<Message>(expected_proto_str_);

  google::protobuf::util::MessageDifferencer differencer;
  std::string differences;
  differencer.ReportDifferencesToString(&differences);
  differencer.set_scope(scope_);

  if (!differencer.Compare(expected_proto, actual_proto)) {
    *listener << "the protos are different:\n" << differences;
    return false;
  }

  return true;
}

// Polymorphic matcher to compare any two protos.
inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const std::string& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

// Polymorphic matcher to compare any two protos.
inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const google::protobuf::Message& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

// Only compare the fields populated in the matcher proto.
template <class InnerProtoMatcher>
inline InnerProtoMatcher Partially(InnerProtoMatcher inner_proto_matcher) {
  inner_proto_matcher.mutable_impl().SetComparePartially();
  return inner_proto_matcher;
}

// Parse input string as a protocol buffer.
template <typename T>
T ParseTextProtoOrDie(const std::string& input) {
  T result;
  REVERB_CHECK(google::protobuf::TextFormat::ParseFromString(input, &result))
      << "Failed to parse: " << input;
  return result;
}

}  // namespace testing
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_TESTING_PROTO_TEST_UTIL_H_
