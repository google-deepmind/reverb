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

#ifndef REVERB_CC_PLATFORM_DEFAULT_HASH_H_
#define REVERB_CC_PLATFORM_DEFAULT_HASH_H_

#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/hash.h"

namespace deepmind {
namespace reverb {
namespace internal {

// The hash of an object of type T is computed by using ConsistentHash.
template <class T, class E = void>
struct HashEq {
  using Hash = tensorflow::hash<T>;
  using Eq = std::equal_to<T>;
};

struct StringHash {
  using is_transparent = void;

  size_t operator()(absl::string_view v) const {
    return tensorflow::hash<absl::string_view>{}(v);
  }

  // TODO(b/173569624): Use tensorflow::hash<absl::Cord> when available.
  size_t operator()(const absl::Cord& v) const {
    tensorflow::hash<absl::string_view> hasher;
    size_t h = hasher("");
    for (auto sv : v.Chunks()) {
      h = tensorflow::Hash64Combine(h, hasher(sv));
    }
    return h;
  }
};

// Supports heterogeneous lookup for string-like elements.
struct StringHashEq {
  using Hash = StringHash;
  struct Eq {
    using is_transparent = void;
    bool operator()(absl::string_view lhs, absl::string_view rhs) const {
      return lhs == rhs;
    }
    bool operator()(const absl::Cord& lhs, const absl::Cord& rhs) const {
      return lhs == rhs;
    }
    bool operator()(const absl::Cord& lhs, absl::string_view rhs) const {
      return lhs == rhs;
    }
    bool operator()(absl::string_view lhs, const absl::Cord& rhs) const {
      return lhs == rhs;
    }
  };
};

template <>
struct HashEq<std::string> : StringHashEq {};
template <>
struct HashEq<absl::string_view> : StringHashEq {};
template <>
struct HashEq<absl::Cord> : StringHashEq {};

// Supports heterogeneous lookup for pointers and smart pointers.
template <class T>
struct HashEq<T*> {
  struct Hash {
    using is_transparent = void;
    template <class U>
    size_t operator()(const U& ptr) const {
      return tensorflow::hash<const T*>{}(HashEq::ToPtr(ptr));
    }
  };
  struct Eq {
    using is_transparent = void;
    template <class A, class B>
    bool operator()(const A& a, const B& b) const {
      return HashEq::ToPtr(a) == HashEq::ToPtr(b);
    }
  };

 private:
  static const T* ToPtr(const T* ptr) { return ptr; }
  template <class U, class D>
  static const T* ToPtr(const std::unique_ptr<U, D>& ptr) {
    return ptr.get();
  }
  template <class U>
  static const T* ToPtr(const std::shared_ptr<U>& ptr) {
    return ptr.get();
  }
};

template <class T, class D>
struct HashEq<std::unique_ptr<T, D>> : HashEq<T*> {};

template <class T>
struct HashEq<std::shared_ptr<T>> : HashEq<T*> {};

}  // namespace internal
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_PLATFORM_DEFAULT_HASH_H_
