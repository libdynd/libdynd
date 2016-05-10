//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <algorithm>
#include <cstring>
#include <map>
#include <new>

#include <dynd/visibility.hpp>

namespace dynd {

template <typename PrefixType, typename DerivedType>
class storagebuf {
protected:
  // Pointer to the kernel function pointers + data
  char *m_data;
  intptr_t m_capacity;
  intptr_t m_size;

  // When the amount of data is small, this static data is used,
  // otherwise dynamic memory is allocated when it gets too big
  char m_static_data[16 * 8];

  bool using_static_data() const { return m_data == &m_static_data[0]; }

public:
  storagebuf() : m_data(m_static_data), m_capacity(sizeof(m_static_data)), m_size(0) {
    set(m_static_data, 0, sizeof(m_static_data));
  }

  ~storagebuf() {
    if (!using_static_data() && m_data != NULL) {
      free(m_data);
    }
  }

  size_t size() const { return m_size; }

  size_t capacity() const { return m_capacity; }

  /**
   * This function ensures that the ckernel's data
   * is at least the required number of bytes. It
   * should only be called during the construction phase
   * of the kernel when constructing a leaf kernel.
   */
  void reserve(intptr_t requested_capacity) {
    if (m_capacity < requested_capacity) {
      // Grow by a factor of 1.5
      // https://github.com/facebook/folly/blob/master/folly/docs/FBVector.md
      intptr_t grown_capacity = m_capacity * 3 / 2;
      if (requested_capacity < grown_capacity) {
        requested_capacity = grown_capacity;
      }
      // Do a realloc
      char *new_data = reinterpret_cast<char *>(realloc(m_data, m_capacity, requested_capacity));
      if (new_data == NULL) {
        reinterpret_cast<DerivedType *>(this)->destroy();
        m_data = NULL;
        throw std::bad_alloc();
      }
      // Zero out the newly allocated capacity
      set(reinterpret_cast<char *>(new_data) + m_capacity, 0, requested_capacity - m_capacity);
      m_data = new_data;
      m_capacity = requested_capacity;
    }
  }

  void *alloc(size_t size) { return std::malloc(size); }

  void *realloc(void *ptr, size_t old_size, size_t new_size) {
    if (using_static_data()) {
      // If we were previously using the static data, do a malloc
      void *new_data = alloc(new_size);
      // If the allocation succeeded, copy the old data as the realloc would
      if (new_data != NULL) {
        copy(new_data, ptr, old_size);
      }
      return new_data;
    } else {
      return std::realloc(ptr, new_size);
    }
  }

  void free(void *ptr) {
    if (!using_static_data()) {
      std::free(ptr);
    }
  }

  void *copy(void *dst, const void *src, size_t size) { return std::memcpy(dst, src, size); }

  void *set(void *dst, int value, size_t size) { return std::memset(dst, value, size); }

  PrefixType *get() const { return reinterpret_cast<PrefixType *>(m_data); }

  /**
   * For use during construction, gets the ckernel component
   * at the requested offset.
   */
  template <typename U>
  U *get_at(size_t offset) {
    return reinterpret_cast<U *>(m_data + offset);
  }

  /**
   * Aligns a size as required by kernels.
   */
  static constexpr size_t aligned_size(size_t size) {
    return (size + static_cast<size_t>(7)) & ~static_cast<size_t>(7);
  }

  /**
   * Creates the kernel, and increments ``m_size`` to the position after it.
   */
  template <typename KernelType, typename... ArgTypes>
  void emplace_back(ArgTypes &&... args) {
    /* Alignment requirement of the type. */
    static_assert(alignof(KernelType) <= 8, "kernel types require alignment to be at most 8 bytes");

    size_t offset = m_size;
    m_size += aligned_size(sizeof(KernelType));
    reserve(m_size);

    KernelType::init(this->get_at<KernelType>(offset), std::forward<ArgTypes>(args)...);
  }

  template <typename KernelType, typename... ArgTypes>
  void emplace_back_sep(ArgTypes &&... args) {
    /* Alignment requirement of the type. */
    static_assert(alignof(KernelType) <= 8, "kernel types require alignment to be at most 8 bytes");

    size_t offset = m_size;
    m_size += aligned_size(sizeof(PrefixType)) + aligned_size(sizeof(KernelType));
    reserve(m_size);

    PrefixType::template init<KernelType>(this->get_at<PrefixType>(offset), std::forward<ArgTypes>(args)...);
  }

  void emplace_back(size_t size) {
    m_size += aligned_size(size);
    reserve(m_size);
  }
};

} // namespace dynd
