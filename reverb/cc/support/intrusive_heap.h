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

// A heap that supports removing and adjusting the weights of arbitrary
// elements.  To do so, it records the heap location of each element in
// a field within that element.
//
// The default IntrusiveHeap uses a heap value field embedded in each item to
// maintain the heap ordering. The item type T provides a public field
// named "heap" (by default) of type IntrusiveHeapLink, which the
// heap uses for this. See intrusive_heap_test.cc for an example.
//
// The storage for the heap link in elements can be customized by providing a
// LinkAccess policy. This should not commonly be required.

#ifndef REVERB_CC_SUPPORT_INTRUSIVE_HEAP_H_
#define REVERB_CC_SUPPORT_INTRUSIVE_HEAP_H_

#include <stddef.h>

#include <memory>
#include <vector>

#include "reverb/cc/platform/logging.h"

namespace deepmind {
namespace reverb {

// The bookkeeping area inside each element, used by IntrusiveHeap.
// IntrusiveHeap objects are configured with a LinkAccess policy with
// read-write access to the IntrusiveHeapLink object within each element.
// Currently implemented as a vector index.
class IntrusiveHeapLink {
 public:
  using size_type = size_t;
  static const size_type kNotMember = -1;

  IntrusiveHeapLink() = default;

  // Only IntrusiveHeap and LinkAccess objects should make these objects.
  explicit IntrusiveHeapLink(size_type pos) : pos_{pos} {}

  // Only IntrusiveHeap and LinkAccess should get the value.
  size_type get() const { return pos_; }

 private:
  size_type pos_{kNotMember};
};

// Manipulate a link object accessible as a data member.
// Usable as an IntrusiveHeap's LinkAccess policy object (see IntrusiveHeap).
template <typename T, IntrusiveHeapLink T::*M>
struct IntrusiveHeapDataMemberLinkAccess {
  IntrusiveHeapLink Get(const T* elem) const { return elem->*M; }
  void Set(T* elem, IntrusiveHeapLink link) const { elem->*M = link; }
};

// The default LinkAccess object, uses the 'heap' data member as a Link.
// Usable as an IntrusiveHeap's LinkAccess policy object (see IntrusiveHeap).
template <typename T>
struct DefaultIntrusiveHeapLinkAccess {
  IntrusiveHeapLink Get(const T* elem) const { return elem->heap; }
  void Set(T* elem, IntrusiveHeapLink link) const { elem->heap = link; }
};

// IntrusiveHeap<T, PtrCompare, LinkAccess, Alloc>
//
//   A min-heap (under PtrCompare ordering) of pointers to T.
//
//   Supports random access removal of elements (in O(lg(N) time), but
//   requires that the pointed-to elements provide a
//   IntrusiveHeapLink data member (usually called 'heap').
//
//   T: the value type to be referenced by the IntrusiveHeap. Note that
//      IntrusiveHeap does not take ownership of its elements; it merely points
//      to them.
//   PtrCompare: a binary predicate applying a strict weak ordering over
//      'const T*' returning true if and only if 'a' should be considered
//      less than 'b'.  Note that IntrusiveHeap is a min-heap under the
//      PtrCompare ordering, such that if PtrCompare(a, b), then 'a' will be
//      popped before 'b'.
//   LinkAccess: Rarely specified, as the default is sufficient for most
//      uses.  A policy class providing functions with the signatures
//      'IntrusiveHeapLink Get(const T* elem)' and
//      void Set(T* elem, IntrusiveHeapLink link)'.
//      These functions allow for customization of location of
//      the IntrusiveHeapLink member in a T* object. The default
//      LinkAccessor policy's Get(elem) and Set(link,elem) functions
//      manipulate the member accessed by 'elem->heap'.
//   Alloc: an STL allocator for T* elements. Default is std::allocator<T*>.
//
//   Note that the IntrusiveHeap does not hold or own any T objects,
//   only pointers to them. Users must manage storage on their own.
template <typename T, typename PtrCompare,
          typename LinkAccess = DefaultIntrusiveHeapLinkAccess<T>,
          typename Alloc = std::allocator<T*> >
class IntrusiveHeap {
 public:
  typedef typename IntrusiveHeapLink::size_type size_type;
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef PtrCompare pointer_compare_type;
  typedef LinkAccess link_access_type;
  typedef Alloc allocator_type;

  explicit IntrusiveHeap(
      const pointer_compare_type& comp = pointer_compare_type(),
      const link_access_type& link_access = link_access_type(),
      const allocator_type& alloc = allocator_type())
      : rep_(comp, link_access, alloc) { }

  size_type size() const {
    return heap().size();
  }

  bool empty() const {
    return heap().empty();
  }

  // Return the top element, but don't remove it.
  pointer top() const {
    REVERB_CHECK(!empty());
    return heap()[0];
  }

  // Remove the top() pointer from the heap and return it.
  pointer Pop() {
    pointer t = top();
    Remove(t);
    return t;
  }

  // Insert 't' into the heap.
  void Push(pointer t) {
    SetPositionOf(t, heap().size());
    heap().push_back(t);
    FixHeapUp(t);
  }

  // Adjust the heap to accommodate changes in '*t'.
  void Adjust(pointer t) {
    REVERB_CHECK(Contains(t));
    size_type h = GetPositionOf(t);
    if (h != 0 && compare()(t, heap()[(h - 1) >> 1])) {
      FixHeapUp(t);
    } else {
      FixHeapDown(t);
    }
  }

  // Remove the specified pointer from the heap.
  void Remove(pointer t) {
    REVERB_CHECK(Contains(t));
    size_type h = GetPositionOf(t);
    SetPositionOf(t, IntrusiveHeapLink::kNotMember);
    if (h == heap().size() - 1) {
      // Fast path for removing from back of heap.
      heap().pop_back();
      return;
    }
    // Move the element from the back of the heap to overwrite 't'.
    pointer& elem = heap()[h];
    elem = heap().back();
    SetPositionOf(elem, h);  // Element has moved, so update its link.
    heap().pop_back();
    Adjust(elem);  // Restore the heap invariant.
  }

  void Clear() {
    heap().clear();
  }

  bool Contains(const_pointer t) const {
    size_type h = GetPositionOf(t);
    return (h != IntrusiveHeapLink::kNotMember) &&
           (h < size()) &&
           heap()[h] == t;
  }

  void reserve(size_type n) { heap().reserve(n); }

  size_type capacity() const { return heap().capacity(); }

  allocator_type get_allocator() const { return rep_.heap_.get_allocator(); }

 private:
  typedef std::vector<pointer, allocator_type> heap_type;

  // Empty base class optimization for pointer_compare and link_access.
  // The heap_ data member retains a copy of the allocator, so it is not
  // stored explicitly.
  struct Rep : pointer_compare_type, link_access_type {
    explicit Rep(const pointer_compare_type& cmp,
                 const link_access_type& link_access,
                 const allocator_type& alloc)
        : pointer_compare_type(cmp),
          link_access_type(link_access),
          heap_(alloc) { }
    heap_type heap_;
  };

  const pointer_compare_type& compare() const { return rep_; }

  pointer_compare_type compare() { return rep_; }

  const link_access_type& link_access() const { return rep_; }

  const heap_type& heap() const { return rep_.heap_; }
  heap_type& heap() { return rep_.heap_; }

  size_type GetPositionOf(const_pointer t) const {
    return link_access().Get(t).get();
  }

  void SetPositionOf(pointer t, size_type pos) const {
    return link_access().Set(t, IntrusiveHeapLink(pos));
  }

  void FixHeapUp(pointer t) {
    size_type h = GetPositionOf(t);
    while (h != 0) {
      size_type parent = (h - 1) >> 1;
      if (compare()(heap()[parent], t)) {
        break;
      }
      heap()[h] = heap()[parent];
      SetPositionOf(heap()[h], h);
      h = parent;
    }
    heap()[h] = t;
    SetPositionOf(t, h);
  }

  void FixHeapDown(pointer t) {
    size_type h = GetPositionOf(t);
    for (;;) {
      size_type kid = (h << 1) + 1;
      if (kid >= heap().size()) {
        break;
      }
      if (kid + 1 < heap().size() &&
          compare()(heap()[kid + 1], heap()[kid])) {
        ++kid;
      }
      if (compare()(t, heap()[kid])) {
        break;
      }
      heap()[h] = heap()[kid];
      SetPositionOf(heap()[h], h);
      h = kid;
    }

    heap()[h] = t;
    SetPositionOf(t, h);
  }

  Rep rep_;
};

}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_SUPPORT_INTRUSIVE_HEAP_H_
