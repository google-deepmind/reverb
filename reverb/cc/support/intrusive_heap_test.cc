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

#include "reverb/cc/support/intrusive_heap.h"

#include <algorithm>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"

namespace deepmind {
namespace reverb {
namespace {

static const int kNumElems = 100;

class IntrusiveHeapTest : public testing::Test {
 protected:
  struct Elem {
    int32_t  val;
    int    iota;
    IntrusiveHeapLink heap;  // position in the heap
  };

  struct ElemChild : Elem {};

  struct ElemLess {
    bool operator()(const Elem* e1, const Elem* e2) const {
      if (e1->val != e2->val) {
        return e1->val < e2->val;
      }
      return e1->iota < e2->iota;
    }
  };
  struct ElemValLess {
    bool operator()(const Elem& e1, const Elem& e2) const {
      return ElemLess()(&e1, &e2);
    }
  };
  struct StatefulLess {
    bool operator()(const Elem* e1, const Elem* e2) const {
      return ElemLess()(e1, e2);
    }
    void* dummy;
  };
  struct StatefulLinkAccess {
    typedef IntrusiveHeapLink Link;
    Link Get(const Elem* e) const { return e->heap; }
    void Set(Elem* e, Link link) const { e->heap = link; }
    void* dummy;
  };

  typedef IntrusiveHeap<Elem, ElemLess> ElemHeap;

  absl::BitGen rnd_;
  ElemHeap      heap_;        // The heap
  std::vector<Elem>  elems_;       // Storage for items in the heap
  std::vector<Elem>  expected_;    // Copy of the elements, for reference

  IntrusiveHeapTest() {}

  // Build a heap.
  void BuildHeap() {
    elems_.resize(kNumElems);
    for (int i = 0; i < kNumElems; i++) {
      elems_[i].val = absl::Uniform<uint32_t>(rnd_);
      elems_[i].iota = i;
      heap_.Push(&elems_[i]);
      expected_.push_back(elems_[i]);
    }
  }

  // Pop the elements from the heap, verifying they are as expected.
  void VerifyHeap() {
    EXPECT_EQ(expected_.size(), heap_.size());
    EXPECT_FALSE(heap_.empty());

    ElemValLess less;
    std::sort(expected_.begin(), expected_.end(), less);

    for (int i = 0; i < expected_.size(); i++) {
      ASSERT_FALSE(heap_.empty());
      Elem* e = heap_.Pop();
      EXPECT_EQ(expected_[i].iota, e->iota) << i;
      EXPECT_EQ(expected_[i].val, e->val) << i;
    }

    EXPECT_EQ(0, heap_.size());
    EXPECT_TRUE(heap_.empty());
  }
};

TEST_F(IntrusiveHeapTest, PushPop) {
  BuildHeap();
  VerifyHeap();
}

TEST_F(IntrusiveHeapTest, Clear) {
  Elem dummy;
  dummy.val = 8675309;
  dummy.iota = 123456;
  heap_.Push(&dummy);
  heap_.Clear();
  EXPECT_EQ(0, heap_.size());
}

TEST_F(IntrusiveHeapTest, Contains) {
  Elem dummy;
  dummy.val = 8675309;
  dummy.iota = 123456;
  EXPECT_FALSE(heap_.Contains(&dummy));
  heap_.Push(&dummy);
  EXPECT_TRUE(heap_.Contains(&dummy));
  heap_.Clear();
  EXPECT_FALSE(heap_.Contains(&dummy));
}

TEST_F(IntrusiveHeapTest, ContainsTwoHeaps) {
  Elem dummy1;
  dummy1.val = 8675309;
  dummy1.iota = 123456;
  Elem dummy2 = dummy1;
  heap_.Push(&dummy1);

  ElemHeap other_heap;

  EXPECT_FALSE(other_heap.Contains(&dummy1));
  EXPECT_FALSE(other_heap.Contains(&dummy2));

  other_heap.Push(&dummy2);

  EXPECT_TRUE(heap_.Contains(&dummy1));
  EXPECT_FALSE(heap_.Contains(&dummy2));
  EXPECT_FALSE(other_heap.Contains(&dummy1));
  EXPECT_TRUE(other_heap.Contains(&dummy2));
}

TEST_F(IntrusiveHeapTest, Remove) {
  BuildHeap();

  // Remove the second half of the elements.
  for (int i = kNumElems / 2; i < kNumElems; i++) {
    heap_.Remove(&elems_[i]);
  }
  expected_.resize(kNumElems / 2);

  VerifyHeap();
}

TEST_F(IntrusiveHeapTest, Adjust) {
  BuildHeap();

  // Adjust the weights of all elements.
  for (int i = 0; i < kNumElems; i++) {
    elems_[i].val = absl::Uniform<uint32_t>(rnd_);
    expected_[i].val = elems_[i].val;
    heap_.Adjust(&elems_[i]);
  }

  VerifyHeap();
}

TEST_F(IntrusiveHeapTest, EmptyBaseClassOptimization) {
  // EBC optimization reduces size from 32 to 24 bytes.
  // Testing that neither stateless PtrCompare nor stateless
  // StatefulLinkAccess contribute to object size.
  EXPECT_LT(sizeof(IntrusiveHeap<Elem, ElemLess>),
            sizeof(IntrusiveHeap<Elem, StatefulLess>));
  EXPECT_LT(sizeof(IntrusiveHeap<Elem, ElemLess>),
            sizeof(IntrusiveHeap<Elem, ElemLess, StatefulLinkAccess>));
  EXPECT_LT(sizeof(IntrusiveHeap<Elem, StatefulLess>),
            sizeof(IntrusiveHeap<Elem, StatefulLess, StatefulLinkAccess>));
}

// Test that an IntrusiveHeap<T> can access
// T's HeapLink element even if T inherits it.
// That is, even if IntrusiveHeapLink data member comes from a base
// class of the Element type, we should still find it.
TEST_F(IntrusiveHeapTest, InheritElement) {
  std::vector<ElemChild> elems(5);
  for (int i = 0; i < elems.size(); ++i) {
    elems[i].val = (i * 19) % 7;
    elems[i].iota = i;
  }
  typedef IntrusiveHeap<ElemChild, ElemLess> Heap;
  Heap heap;
  for (ElemChild& e : elems) {
    heap.Push(&e);
  }
  std::vector<ElemChild> expected = elems;
  std::sort(expected.begin(), expected.end(), ElemValLess());
  std::vector<ElemChild> actual;
  while (!heap.empty()) {
    actual.push_back(*heap.Pop());
  }
  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(expected[i].val, actual[i].val);
  }
}

class SyntheticLinkTest : public testing::Test {
 public:
  struct Element {
    std::string value;
    unsigned char heap;
  };
  struct Access {
    using Link = IntrusiveHeapLink;
    Link Get(const Element* elem) const {
      return Link(elem->heap);
    }
    void Set(Element* elem, Link link) const {
      elem->heap = link.get();
    }
  };
  struct PtrCompare {
    bool operator()(const Element* a, const Element* b) const {
      return a->value < b->value;
    }
  };
};

TEST_F(SyntheticLinkTest, SetAndGet) {
  IntrusiveHeap<Element, PtrCompare, Access> heap;
  std::vector<Element> elems{{"d"}, {"b"}, {"e"}, {"a"}, {"c"}};
  for (auto& e : elems) heap.Push(&e);
  std::vector<std::string> out;
  while (!heap.empty()) {
    out.push_back(heap.top()->value);
    heap.Pop();
  }
  auto sorted = out;
  std::sort(sorted.begin(), sorted.end());
  EXPECT_THAT(out, testing::ElementsAreArray(sorted));
}

TEST_F(SyntheticLinkTest, ReserveAndCapacity) {
  IntrusiveHeap<Element, PtrCompare, Access> heap;
  std::vector<Element> elems{{"d"}, {"b"}, {"e"}, {"a"}, {"c"}};
  EXPECT_EQ(0, heap.capacity());
  EXPECT_EQ(0, heap.size());
  heap.reserve(elems.size());
  EXPECT_EQ(0, heap.size());
  EXPECT_GE(heap.capacity(), elems.size());
  for (auto& e : elems) heap.Push(&e);
  EXPECT_EQ(heap.size(), elems.size());
  EXPECT_GE(heap.capacity(), elems.size());
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind
