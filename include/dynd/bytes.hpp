//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <algorithm>

#include <dynd/config.hpp>

namespace dynd {

class DYND_API bytes {
protected:
  char *m_data;
  size_t m_size;

public:
  bytes() : m_data(NULL), m_size(0)
  {
  }

  bytes(const char *data, size_t size) : m_data(new char[size]), m_size(size)
  {
    std::copy_n(data, m_size, m_data);
  }

  bytes(const bytes &other) : m_data(new char[other.m_size]), m_size(other.m_size)
  {
    std::copy_n(other.m_data, m_size, m_data);
  }

  ~bytes()
  {
    delete[] m_data;
  }

  char *data()
  {
    return m_data;
  }

  const char *data() const
  {
    return m_data;
  }

  size_t size() const
  {
    return m_size;
  }

  bytes &assign(const char *data, size_t size)
  {
    if (size != m_size) {
      delete[] m_data;
      m_data = new char[size];
      m_size = size;
    }

    std::copy_n(data, m_size, m_data);

    return *this;
  }

  void clear()
  {
    if (m_size != 0) {
      delete[] m_data;
      m_size = 0;
    }
  }

  void resize(size_t size)
  {
    if (size != m_size) {
      char *data = new char[size];
      std::copy_n(m_data, std::min(size, m_size), data);
      delete[] m_data;

      m_data = data;
      m_size = size;
    }
  }

  char *begin()
  {
    return m_data;
  }

  const char *begin() const
  {
    return m_data;
  }

  char *end()
  {
    return m_data + m_size;
  }

  const char *end() const
  {
    return m_data + m_size;
  }

  bytes &operator=(const bytes &rhs)
  {
    return assign(rhs.m_data, rhs.m_size);
  }
  bool operator==(const bytes &rhs) const
  {
    return m_size == rhs.m_size && std::memcmp(m_data, rhs.m_data, m_size) == 0;
  }

  bool operator!=(const bytes &rhs) const
  {
    return m_size != rhs.m_size || std::memcmp(m_data, rhs.m_data, m_size) != 0;
  }
};

namespace detail {

  class value_bytes {
  protected:
    char *m_data;
    size_t m_size;

    value_bytes(char *data, size_t size) : m_data(data), m_size(size)
    {
    }

  public:
    value_bytes() : m_data(NULL), m_size(0)
    {
    }

    value_bytes(const value_bytes &other) : m_data(new char[other.m_size]), m_size(other.m_size)
    {
      std::copy_n(other.m_data, m_size, m_data);
    }

    ~value_bytes()
    {
      delete[] m_data;
    }

    operator char *()
    {
      return m_data;
    }

    operator const char *() const
    {
      return m_data;
    }

    value_bytes &operator=(const value_bytes &rhs)
    {
      std::copy_n(rhs.m_data, m_size, m_data);

      return *this;
    }
  };

} // namespace dynd::detail

class strided_iterator : public detail::value_bytes,
                         public std::iterator<std::random_access_iterator_tag, detail::value_bytes> {
  intptr_t m_stride;

public:
  strided_iterator() : m_stride(0) {};

  strided_iterator(char *data, size_t size, intptr_t stride) : value_bytes(data, size), m_stride(stride)
  {
  }

  strided_iterator(const strided_iterator &other) : value_bytes(other.m_data, other.m_size), m_stride(other.m_stride)
  {
  }

  ~strided_iterator()
  {
    m_data = NULL;
    m_size = 0;
  }

  intptr_t stride() const
  {
    return m_stride;
  }

  value_bytes &operator*()
  {
    return *this;
  }

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

  strided_iterator &operator+=(int i)
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

  strided_iterator &operator-=(int i)
  {
    m_data -= i * m_stride;
    return *this;
  }

  bool operator<(const strided_iterator &rhs) const
  {
    return m_data < rhs.m_data;
  }

  bool operator<=(const strided_iterator &rhs) const
  {
    return m_data <= rhs.m_data;
  }

  bool operator==(const strided_iterator &rhs) const
  {
    return m_data == rhs.m_data;
  }

  bool operator!=(const strided_iterator &rhs) const
  {
    return m_data != rhs.m_data;
  }

  bool operator>=(const strided_iterator &rhs) const
  {
    return m_data >= rhs.m_data;
  }

  bool operator>(const strided_iterator &rhs) const
  {
    return m_data > rhs.m_data;
  }

  std::ptrdiff_t operator-(strided_iterator rhs)
  {
    return (m_data - rhs.m_data) / m_stride;
  }

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
