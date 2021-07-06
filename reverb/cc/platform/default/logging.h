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

// Copyright 2016-2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
//
// A minimal replacement for "glog"-like functionality. Does not provide output
// in a separate thread nor backtracing.

#ifndef REVERB_CC_PLATFORM_DEFAULT_LOGGING_H_
#define REVERB_CC_PLATFORM_DEFAULT_LOGGING_H_

// IWYU pragma: private
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>

#ifdef __GNUC__
#define PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#define PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define NORETURN __attribute__((noreturn))
#else
#define PREDICT_TRUE(x) (x)
#define PREDICT_FALSE(x) (x)
#define NORETURN
#endif

namespace deepmind {
namespace reverb {
namespace internal {

struct CheckOpString {
  explicit CheckOpString(std::string* str) : str_(str) {}
  explicit operator bool() const { return PREDICT_FALSE(str_ != nullptr); }
  std::string* const str_;
};

template <typename T1, typename T2>
CheckOpString MakeCheckOpString(const T1& v1, const T2& v2,
                                const char* exprtext) {
  std::ostringstream oss;
  oss << exprtext << " (" << v1 << " vs. " << v2 << ")";
  return CheckOpString(new std::string(oss.str()));
}

#define DEFINE_CHECK_OP_IMPL(name, op)                                    \
  template <typename T1, typename T2>                                     \
  inline CheckOpString name##Impl(const T1& v1, const T2& v2,             \
                                  const char* exprtext) {                 \
    if (PREDICT_TRUE(v1 op v2)) {                                         \
      return CheckOpString(nullptr);                                      \
    } else {                                                              \
      return (MakeCheckOpString)(v1, v2, exprtext);                       \
    }                                                                     \
  }                                                                       \
  inline CheckOpString name##Impl(int v1, int v2, const char* exprtext) { \
    return (name##Impl<int, int>)(v1, v2, exprtext);                      \
  }
DEFINE_CHECK_OP_IMPL(Check_EQ, ==)
DEFINE_CHECK_OP_IMPL(Check_NE, !=)
DEFINE_CHECK_OP_IMPL(Check_LE, <=)
DEFINE_CHECK_OP_IMPL(Check_LT, <)
DEFINE_CHECK_OP_IMPL(Check_GE, >=)
DEFINE_CHECK_OP_IMPL(Check_GT, >)
#undef DEFINE_CHECK_OP_IMPL

class LogMessage {
 public:
  LogMessage(const char* file, int line) {
    std::clog << "[" << file << ":" << line << "] ";
  }

  ~LogMessage() { std::clog << "\n"; }

  std::ostream& stream() && { return std::clog; }
};

class LogMessageFatal {
 public:
  LogMessageFatal(const char* file, int line) {
    stream_ << "[" << file << ":" << line << "] ";
  }

  LogMessageFatal(const char* file, int line, const CheckOpString& result) {
    stream_ << "[" << file << ":" << line << "] Check failed: " << *result.str_;
  }

  ~LogMessageFatal() NORETURN;

  std::ostream& stream() && { return stream_; }

 private:
  std::ostringstream stream_;
};

inline LogMessageFatal::~LogMessageFatal() {
  std::cerr << stream_.str() << std::endl;
  std::abort();
}

struct NullStream {};

template <typename T>
NullStream&& operator<<(NullStream&& s, T&&) {
  return std::move(s);
}

enum class LogSeverity {
  kFatal,
  kNonFatal,
};

LogMessage LogStream(
    std::integral_constant<LogSeverity, LogSeverity::kNonFatal>);
LogMessageFatal LogStream(
    std::integral_constant<LogSeverity, LogSeverity::kFatal>);

struct Voidify {
  void operator&(std::ostream&) {}
};

class LogEveryNState {
 public:
  bool ShouldLog(int n);
  uint32_t counter() { return counter_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32_t> counter_{0};
};

class LogFirstNState {
 public:
  bool ShouldLog(int n);
  uint32_t counter() { return counter_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32_t> counter_{0};
};

class LogEveryPow2State {
 public:
  bool ShouldLog(int ignored);
  uint32_t counter() { return counter_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint32_t> counter_{0};
};

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind

#define REVERB_CHECK_OP_LOG(name, op, val1, val2, log)         \
  while (::deepmind::reverb::internal::CheckOpString _result = \
             ::deepmind::reverb::internal::name##Impl(         \
                 val1, val2, #val1 " " #op " " #val2))         \
  log(__FILE__, __LINE__, _result).stream()

#define REVERB_CHECK_OP(name, op, val1, val2) \
  REVERB_CHECK_OP_LOG(name, op, val1, val2,   \
                      ::deepmind::reverb::internal::LogMessageFatal)

#define REVERB_CHECK_EQ(val1, val2) REVERB_CHECK_OP(Check_EQ, ==, val1, val2)
#define REVERB_CHECK_NE(val1, val2) REVERB_CHECK_OP(Check_NE, !=, val1, val2)
#define REVERB_CHECK_LE(val1, val2) REVERB_CHECK_OP(Check_LE, <=, val1, val2)
#define REVERB_CHECK_LT(val1, val2) REVERB_CHECK_OP(Check_LT, <, val1, val2)
#define REVERB_CHECK_GE(val1, val2) REVERB_CHECK_OP(Check_GE, >=, val1, val2)
#define REVERB_CHECK_GT(val1, val2) REVERB_CHECK_OP(Check_GT, >, val1, val2)

#define REVERB_QCHECK_EQ(val1, val2) REVERB_CHECK_OP(Check_EQ, ==, val1, val2)
#define REVERB_QCHECK_NE(val1, val2) REVERB_CHECK_OP(Check_NE, !=, val1, val2)
#define REVERB_QCHECK_LE(val1, val2) REVERB_CHECK_OP(Check_LE, <=, val1, val2)
#define REVERB_QCHECK_LT(val1, val2) REVERB_CHECK_OP(Check_LT, <, val1, val2)
#define REVERB_QCHECK_GE(val1, val2) REVERB_CHECK_OP(Check_GE, >=, val1, val2)
#define REVERB_QCHECK_GT(val1, val2) REVERB_CHECK_OP(Check_GT, >, val1, val2)

#define REVERB_CHECK(condition)                                              \
  while (auto _result = ::deepmind::reverb::internal::CheckOpString(         \
             (condition) ? nullptr : new std::string(#condition)))           \
  ::deepmind::reverb::internal::LogMessageFatal(__FILE__, __LINE__, _result) \
      .stream()

#define REVERB_QCHECK(condition) REVERB_CHECK(condition)

#define REVERB_FATAL ::deepmind::reverb::internal::LogSeverity::kFatal
#define REVERB_QFATAL ::deepmind::reverb::internal::LogSeverity::kFatal
#define REVERB_INFO ::deepmind::reverb::internal::LogSeverity::kNonFatal
#define REVERB_WARNING ::deepmind::reverb::internal::LogSeverity::kNonFatal
#define REVERB_ERROR ::deepmind::reverb::internal::LogSeverity::kNonFatal

#define REVERB_LOG(level)                                               \
  decltype(::deepmind::reverb::internal::LogStream(                     \
      std::integral_constant<::deepmind::reverb::internal::LogSeverity, \
                             level>()))(__FILE__, __LINE__)             \
      .stream()

#define REVERB_VLOG(level) ::deepmind::reverb::internal::NullStream()

#define REVERB_LOG_IF(level, condition)                                    \
  !(condition)                                                             \
      ? static_cast<void>(0)                                               \
      : ::deepmind::reverb::internal::Voidify() &                          \
            decltype(::deepmind::reverb::internal::LogStream(              \
                std::integral_constant<                                    \
                    ::deepmind::reverb::internal::LogSeverity, level>()))( \
                __FILE__, __LINE__)                                        \
                .stream()

#define REVERB_LOGGING_INTERNAL_STATEFUL_CONDITION(kind, condition, arg)   \
  for (bool logging_internal_stateful_condition_do_log(condition);  \
       logging_internal_stateful_condition_do_log;                  \
       logging_internal_stateful_condition_do_log = false)          \
    for (static ::deepmind::reverb::internal::Log##kind##State      \
             logging_internal_stateful_condition_state;             \
         logging_internal_stateful_condition_do_log &&              \
         logging_internal_stateful_condition_state.ShouldLog(arg);  \
         logging_internal_stateful_condition_do_log = false)        \
      for (const uint32_t COUNTER ABSL_ATTRIBUTE_UNUSED =           \
               logging_internal_stateful_condition_state.counter(); \
           logging_internal_stateful_condition_do_log;              \
           logging_internal_stateful_condition_do_log = false)

#define REVERB_LOG_EVERY_N(level, n)                          \
  REVERB_LOGGING_INTERNAL_STATEFUL_CONDITION(EveryN, true, n) \
  REVERB_LOG(level)

#define REVERB_LOG_FIRST_N(level, n)                          \
  REVERB_LOGGING_INTERNAL_STATEFUL_CONDITION(FirstN, true, n) \
  REVERB_LOG(level)

#define REVERB_LOG_EVERY_POW_2(level)                            \
  REVERB_LOGGING_INTERNAL_STATEFUL_CONDITION(EveryPow2, true, 0) \
  REVERB_LOG(level)

#endif  // REVERB_CC_PLATFORM_DEFAULT_LOGGING_H_
