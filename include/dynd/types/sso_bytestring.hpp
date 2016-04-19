//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

namespace dynd {

/**
 * An implementation of an SSO bytestring, used as a base class for both nd::string and nd::bytes with different
 * template parameter options to include or exclude a NUL terminator, respectively.
 *
 * The template parameter `NulPadding` must be either 0 or 1, nd::bytes uses 0 and nd::string uses 1.
 *
 * The overall strategy of the implementation is to provide an internal `is_sso()` function to identify whether storage
 * is using SSO, then have code paths that use the `sso_*` and `heap_*` functions to do their things with no additional
 * checking for whether SSO is active.
 */
template <size_t NulPadding>
class sso_bytestring {
protected:
  int64_t m_pointer;
  int64_t m_size;

  /** Whether the string is using SSO or allocated memory */
  bool is_sso() const { return m_size >= 0; }

  /** When SSO is used, the size is stored in the last byte */
  size_t sso_size() const { return static_cast<size_t>(static_cast<uint64_t>(m_size) >> 56); }
  /** When SSO is used, the data pointer inside the object */
  char *sso_data() { return reinterpret_cast<char *>(this); }
  const char *sso_data() const { return reinterpret_cast<const char *>(this); }
  constexpr size_t sso_capacity() const { return 15u - NulPadding; }
  /** When the object has no memory allocated and `size` is <= capacity(), do straight SSO assignment */
  void sso_assign(const char *data, size_t size) {
      m_size = static_cast<int64_t>(static_cast<uint64_t>(size) << 56);
      DYND_MEMCPY(sso_data(), data, size);
      // Zero out the rest of the bytes for a unique representation
      memset(sso_data() + size, 0, 15u - size);
  }

  /** When SSO is not used, the size is stored in m_size */
  size_t heap_size() const { return static_cast<size_t>(~m_size); }
  char *heap_buffer() { return reinterpret_cast<char *>(static_cast<intptr_t>(m_pointer)); }
  const char *heap_buffer() const { return reinterpret_cast<const char *>(static_cast<intptr_t>(m_pointer)); }
  /** When SSO is not used, the data pointer after a size_t in the data buffer */
  char *heap_data() { return heap_buffer() + sizeof(size_t); }
  const char *heap_data() const { return heap_buffer() + sizeof(size_t); }
  /** When SSO is not used, the capacity is stored at the start of the data buffer */
  size_t heap_capacity() const { return *reinterpret_cast<const size_t *>(heap_buffer()); }
  /**
   * When the object has no memory allocated straight heap assignment overwriting existing data.
   * NOTE: If it throws (memory allocation failure), it hasn't written into `this`.
   */
  void heap_assign(const char *data, size_t size) {
    char *buffer = new char[size + sizeof(size_t) + NulPadding];
    *reinterpret_cast<size_t *>(buffer) = size;
    DYND_MEMCPY(buffer + sizeof(size_t), data, size);
    if (NulPadding) {
      buffer[sizeof(size_t) + size] = 0;
    }
    m_pointer = reinterpret_cast<intptr_t>(buffer);
    m_size = ~static_cast<int64_t>(size);
  }

public:
  /** Default-constructed empty bytestring */
  sso_bytestring() : m_pointer(0), m_size(0) {}

  sso_bytestring(const sso_bytestring &rhs) {
    if (rhs.size() <= sso_capacity()) {
      sso_assign(rhs.data(), rhs.size());
    } else {
      heap_assign(rhs.data(), rhs.size());
    }
  }

  sso_bytestring(sso_bytestring &&rhs) {
    m_pointer = rhs.m_pointer;
    m_size = rhs.m_size;
    rhs.m_pointer = 0;
    rhs.m_size = 0;
  }

  sso_bytestring(const char *data, size_t size) {
    if (size <= sso_capacity()) {
      sso_assign(data, size);
    } else {
      heap_assign(data, size);
    }
  }

  ~sso_bytestring() {
    if (!is_sso()) {
      delete[] heap_buffer();
    }
  }

  /** The size of the string in bytes */
  size_t size() const { return is_sso() ? sso_size() : heap_size(); }

  /** The current capacity of the string in bytes, excluding any NUL padding */
  size_t capacity() const { return is_sso() ? sso_capacity() : heap_capacity(); }

  char *data() { return is_sso() ? sso_data() : heap_data(); }
  const char *data() const { return is_sso() ? sso_data() : heap_data(); }

  /** Assigns the provided byte string by value */
  void assign(const char *bytestr, size_t size) {
    if (is_sso()) {
      if (size <= sso_capacity()) {
        sso_assign(bytestr, size);
      } else {
        heap_assign(bytestr, size);
      }
    } else if (size <= capacity()) {
      char *heapdata = heap_data();
      DYND_MEMCPY(heapdata, bytestr, size);
      if (NulPadding) {
        heapdata[size] = 0;
      }
      m_size = ~static_cast<int64_t>(size);
    } else {
      char *buffer = heap_buffer();
      heap_assign(bytestr, size);
      delete[] buffer;
    }
  }

  sso_bytestring &operator=(const sso_bytestring &rhs) {
    assign(rhs.data(), rhs.size());
    return *this;
  }

  sso_bytestring &operator=(sso_bytestring &&rhs) {
    if (!is_sso()) {
      delete[] heap_buffer();
    }
    m_pointer = rhs.m_pointer;
    m_size = rhs.m_size;
    rhs.m_pointer = 0;
    rhs.m_size = 0;
    return *this;
  }

  void append(const char *bytestr, size_t bssize) {
    size_t old_size = size();
    resize_grow(old_size + bssize);
    DYND_MEMCPY(data() + old_size, bytestr, bssize);
  }

  void clear() {
    if (!is_sso()) {
      delete[] heap_buffer();
    }
    m_pointer = 0;
    m_size = 0;
  }

  /** If necessary, allocate memory so that the internal capacity is as requested */
  void reserve(size_t new_capacity) {
    if (capacity() < new_capacity) {
      size_t current_size = size();
      char *new_data = new char[new_capacity + sizeof(size_t) + NulPadding];
      *reinterpret_cast<size_t *>(new_data) = new_capacity;
      DYND_MEMCPY(new_data + sizeof(size_t), data(), current_size + NulPadding);
      if (!is_sso()) {
        delete[] heap_buffer();
      }
      m_size = ~static_cast<int64_t>(current_size);
      m_pointer = reinterpret_cast<intptr_t>(new_data);
    }
  }

  /** Like reserve(), but with exponential growth */
  void reserve_grow(size_t new_capacity) { reserve(std::max(new_capacity, capacity() * 3 / 2)); }

  /** Resizes the bytestring to the specified number of bytes */
  void resize(size_t new_size) {
    reserve(new_size);
    if (is_sso()) {
      m_size = static_cast<int64_t>(static_cast<uint64_t>(m_size) | (static_cast<uint64_t>(new_size) << 56));
      // Always keep the unused SSO bytes as 0 for unique representation and NUL-padding when that is enabled
      memset(sso_data() + new_size, 0, 15u - new_size);
    } else {
      m_size = ~static_cast<int64_t>(new_size);
      if (NulPadding) {
        heap_data()[new_size] = 0;
      }
    }
  }

  /** Resizes the bytestring to the specified number of bytes, using exponential growth if it grows */
  void resize_grow(size_t new_size) {
    reserve_grow(new_size);
    // NOTE: Could duplicate most of resize() to avoid the extra reserve() call there, just doing this for simplicity
    //       right now
    resize(new_size);
  }

  const char *begin() const { return data(); }
  char *begin() { return data(); }
  const char *end() const { return data() + size(); }
  char *end() { return data() + size(); }

  bool empty() const { return size() == 0; }

  bool operator==(const sso_bytestring &rhs) const {
    size_t sz = size();
    return sz == rhs.size() && memcmp(data(), rhs.data(), sz) == 0;
  }

  bool operator!=(const sso_bytestring &rhs) const { return !operator==(rhs); }
};

} // namespace dynd
