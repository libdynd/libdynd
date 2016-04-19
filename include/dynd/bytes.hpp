//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <algorithm>

#include <dynd/config.hpp>
#include <dynd/types/sso_bytestring.hpp>

namespace dynd {

class DYNDT_API bytes : public sso_bytestring<0> {
public:
  bytes() {}

  bytes(const bytes &rhs) : sso_bytestring(rhs) {}

  bytes(const bytes &&rhs) : sso_bytestring(std::move(rhs)) {}

  bytes(const char *bytestr, size_t size) : sso_bytestring(bytestr, size) {}

  template <size_t N>
  bytes(const char (&data)[N]) : sso_bytestring(data, N - 1) {}

  bytes &append(const char *bytestr, size_t size)
  {
    sso_bytestring::append(bytestr, size);
    return *this;
  }

  bytes &assign(const char *bytestr, size_t size)
  {
    sso_bytestring::assign(bytestr, size);
    return *this;
  }

  bytes &operator=(const bytes &rhs) {
    sso_bytestring::assign(rhs.data(), rhs.size());
    return *this;
  }

  bytes &operator=(bytes &&rhs) {
    sso_bytestring::operator=(std::move(rhs));
    return *this;
  }

  bool operator==(const bytes &rhs) const { return sso_bytestring::operator==(rhs); }

  bool operator!=(const bytes &rhs) const { return sso_bytestring::operator!=(rhs); }

  bytes operator+(const bytes &rhs) {
    bytes result;
    result.resize(size() + rhs.size());
    DYND_MEMCPY(result.data(), data(), size());
    DYND_MEMCPY(result.data() + size(), rhs.data(), rhs.size());
    return result;
  }

  bytes &operator+=(const bytes &rhs) {
    sso_bytestring::append(rhs.data(), rhs.size());
    return *this;
  }
};

namespace detail {

  class value_bytes {
  protected:
    char *m_data;
    size_t m_size;

    value_bytes(char *data, size_t size) : m_data(data), m_size(size) {}

  public:
    value_bytes() : m_data(NULL), m_size(0) {}

    value_bytes(const value_bytes &other) : m_data(new char[other.m_size]), m_size(other.m_size)
    {
      memcpy(m_data, other.m_data, m_size);
    }

    ~value_bytes() { delete[] m_data; }

    operator char *() { return m_data; }

    operator const char *() const { return m_data; }

    value_bytes &operator=(const value_bytes &rhs)
    {
      // TODO: This is unsafe, need to fix it.
      memcpy(m_data, rhs.m_data, m_size);

      return *this;
    }
  };

} // namespace dynd::detail

class strided_iterator : public detail::value_bytes,
                         public std::iterator<std::random_access_iterator_tag, detail::value_bytes> {
  intptr_t m_stride;

public:
  strided_iterator() : m_stride(0){};

  strided_iterator(char *data, size_t size, intptr_t stride) : value_bytes(data, size), m_stride(stride) {}

  strided_iterator(const strided_iterator &other) : value_bytes(other.m_data, other.m_size), m_stride(other.m_stride) {}

  ~strided_iterator()
  {
    m_data = NULL;
    m_size = 0;
  }

  intptr_t stride() const { return m_stride; }

  value_bytes &operator*() { return *this; }

  strided_iterator &operator++()
  {
    m_data += m_stride;
    return *this;
  }

  strided_iterator operator++(int)
  {
    strided_iterator tmp(*this);
    operator++();
    return tmp;
  }

  strided_iterator &operator+=(std::ptrdiff_t i)
  {
    m_data += i * m_stride;
    return *this;
  }

  strided_iterator &operator--()
  {
    m_data -= m_stride;
    return *this;
  }

  strided_iterator operator--(int)
  {
    strided_iterator tmp(*this);
    operator--();
    return tmp;
  }

  strided_iterator &operator-=(std::ptrdiff_t i)
  {
    m_data -= i * m_stride;
    return *this;
  }

  bool operator<(const strided_iterator &rhs) const { return m_data < rhs.m_data; }

  bool operator<=(const strided_iterator &rhs) const { return m_data <= rhs.m_data; }

  bool operator==(const strided_iterator &rhs) const { return m_data == rhs.m_data; }

  bool operator!=(const strided_iterator &rhs) const { return m_data != rhs.m_data; }

  bool operator>=(const strided_iterator &rhs) const { return m_data >= rhs.m_data; }

  bool operator>(const strided_iterator &rhs) const { return m_data > rhs.m_data; }

  std::ptrdiff_t operator-(strided_iterator rhs) { return (m_data - rhs.m_data) / m_stride; }

  strided_iterator &operator=(const strided_iterator &other)
  {
    m_data = other.m_data;
    m_size = other.m_size;

    return *this;
  }

  friend strided_iterator operator+(strided_iterator lhs, std::ptrdiff_t rhs);
  friend strided_iterator operator-(strided_iterator lhs, std::ptrdiff_t rhs);
};

inline strided_iterator operator+(strided_iterator lhs, std::ptrdiff_t rhs)
{
  return strided_iterator(lhs.m_data + rhs * lhs.m_stride, lhs.m_size, lhs.m_stride);
}

inline strided_iterator operator-(strided_iterator lhs, std::ptrdiff_t rhs)
{
  return strided_iterator(lhs.m_data - rhs * lhs.m_stride, lhs.m_size, lhs.m_stride);
}

} // namespace dynd
