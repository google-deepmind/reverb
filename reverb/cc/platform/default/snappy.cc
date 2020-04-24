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

#include "reverb/cc/platform/snappy.h"

#include <cstring>

#include "absl/meta/type_traits.h"
#include "snappy-sinksource.h"  // NOLINT(build/include)
#include "snappy.h"  // NOLINT(build/include)

namespace deepmind {
namespace reverb {

namespace {

// Helpers for STLStringResizeUninitialized
// HasMember is true_type or false_type, depending on whether or not
// T has a __resize_default_init member. Resize will call the
// __resize_default_init member if it exists, and will call the resize
// member otherwise.
template <typename string_type, typename = void>
struct ResizeUninitializedTraits {
  using HasMember = std::false_type;
  static void Resize(string_type* s, size_t new_size) { s->resize(new_size); }
};

// __resize_default_init is provided by libc++ >= 8.0.
template <typename string_type>
struct ResizeUninitializedTraits<
    string_type, absl::void_t<decltype(std::declval<string_type&>()
                                           .__resize_default_init(237))> > {
  using HasMember = std::true_type;
  static void Resize(string_type* s, size_t new_size) {
    s->__resize_default_init(new_size);
  }
};

template <typename string_type>
inline constexpr bool STLStringSupportsNontrashingResize(string_type*) {
  return ResizeUninitializedTraits<string_type>::HasMember::value;
}

// Resize string `s` to `new_size`, leaving the data uninitialized.
static inline void STLStringResizeUninitialized(std::string* s,
                                                size_t new_size) {
  ResizeUninitializedTraits<std::string>::Resize(s, new_size);
}

class StringSink : public snappy::Sink {
 public:
  explicit StringSink(std::string* dest) : dest_(dest) {}

  StringSink(const StringSink&) = delete;
  StringSink& operator=(const StringSink&) = delete;

  void Append(const char* data, size_t n) override {
    if (STLStringSupportsNontrashingResize(dest_)) {
      size_t current_size = dest_->size();
      if (data == (const_cast<char*>(dest_->data()) + current_size)) {
        // Zero copy append
        STLStringResizeUninitialized(dest_, current_size + n);
        return;
      }
    }
    dest_->append(data, n);
  }

  char* GetAppendBuffer(size_t size, char* scratch) override {
    if (!STLStringSupportsNontrashingResize(dest_)) {
      return scratch;
    }

    const size_t current_size = dest_->size();
    if ((size + current_size) > dest_->capacity()) {
      // Use resize instead of reserve so that we grow by the strings growth
      // factor. Then reset the size to where it was.
      STLStringResizeUninitialized(dest_, size + current_size);
      STLStringResizeUninitialized(dest_, current_size);
    }

    // If string size is zero, then string_as_array() returns nullptr, so
    // we need to use data() instead
    return const_cast<char*>(dest_->data()) + current_size;
  }

 private:
  std::string* dest_;
};

// TODO(b/140988915): See if this can be moved to snappy's codebase.
class CheckedByteArraySink : public snappy::Sink {
  // A snappy Sink that takes an output buffer and a capacity value.  If the
  // writer attempts to write more data than capacity, it does the safe thing
  // and doesn't attempt to write past the data boundary.  After writing,
  // call sink.Overflowed() to see if an overflow occurred.

 public:
  CheckedByteArraySink(char* outbuf, size_t capacity)
      : outbuf_(outbuf), capacity_(capacity), size_(0), overflowed_(false) {}
  CheckedByteArraySink(const CheckedByteArraySink&) = delete;
  CheckedByteArraySink& operator=(const CheckedByteArraySink&) = delete;

  void Append(const char* bytes, size_t n) override {
    size_t available = capacity_ - size_;
    if (n > available) {
      n = available;
      overflowed_ = true;
    }
    if (n > 0 && bytes != (outbuf_ + size_)) {
      // Catch cases where the pointer returned by GetAppendBuffer() was
      // modified.
      assert(!(outbuf_ <= bytes && bytes < outbuf_ + capacity_));
      memcpy(outbuf_ + size_, bytes, n);
    }
    size_ += n;
  }

  char* GetAppendBuffer(size_t length, char* scratch) override {
    size_t available = capacity_ - size_;
    if (available >= length) {
      return outbuf_ + size_;
    } else {
      return scratch;
    }
  }

  // Returns the number of bytes actually written to the sink.
  size_t NumberOfBytesWritten() const { return size_; }

  // Returns true if any bytes were discarded during the Append(), i.e., if
  // Append() attempted to write more than 'capacity' bytes.
  bool Overflowed() const { return overflowed_; }

 private:
  char* outbuf_;
  const size_t capacity_;
  size_t size_;
  bool overflowed_;
};

}  // namespace

template <>
size_t SnappyCompressFromString(absl::string_view input, std::string* output) {
  snappy::ByteArraySource source(input.data(), input.size());
  StringSink sink(output);
  return snappy::Compress(&source, &sink);
}

template <>
bool SnappyUncompressToString(const std::string& input, size_t output_capacity,
                              char* output) {
  snappy::ByteArraySource source(input.data(), input.size());
  CheckedByteArraySink sink(output, output_capacity);
  return snappy::Uncompress(&source, &sink);
}

}  // namespace reverb
}  // namespace deepmind
