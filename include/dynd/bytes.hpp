//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

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

  bytes(char *data, size_t size) : m_data(data), m_size(size)
  {
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

  void assign(char *data, size_t size)
  {
    m_data = data;
    m_size = size;
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

  bool operator==(const bytes &rhs) const
  {
    return m_size == rhs.m_size && std::memcmp(m_data, rhs.m_data, m_size) == 0;
  }

  bool operator!=(const bytes &rhs) const
  {
    return m_size != rhs.m_size || std::memcmp(m_data, rhs.m_data, m_size) != 0;
  }
};

class std_bytes {
protected:
  char *m_data;
  size_t m_size;

public:
  std_bytes(char *data, size_t size) : m_data(data), m_size(size)
  {
  }

  std_bytes(const std_bytes &other)
  {
    m_data = reinterpret_cast<char *>(malloc(other.m_size));
    m_size = other.m_size;
    std::memcpy(m_data, other.m_data, other.m_size);
  }

  ~std_bytes()
  {
    if (m_data != NULL) {
      free(m_data);
    }
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

  std_bytes &operator=(const std_bytes &rhs)
  {
    std::memcpy(m_data, rhs.m_data, rhs.m_size);
    m_size = rhs.m_size;

    return *this;
  }
};

class bytes_iterator : public std_bytes {
  intptr_t m_stride;

public:
  bytes_iterator(char *data, size_t size, intptr_t stride) : std_bytes(data, size), m_stride(stride)
  {
  }

  bytes_iterator(const bytes_iterator &other) : std_bytes(other.m_data, other.m_size), m_stride(other.m_stride)
  {
  }

  ~bytes_iterator()
  {
    m_data = NULL;
    m_size = 0;
  }

  intptr_t stride() const
  {
    return m_stride;
  }

  std_bytes &operator*()
  {
    return *this;
  }

  bytes_iterator &operator++()
  {
    m_data += m_stride;
    return *this;
  }

  bytes_iterator operator++(int)
  {
    bytes_iterator tmp(*this);
    operator++();
    return tmp;
  }

  bytes_iterator &operator+=(int i)
  {
    m_data += i * m_stride;
    return *this;
  }

  bytes_iterator &operator--()
  {
    m_data -= m_stride;
    return *this;
  }

  bytes_iterator operator--(int)
  {
    bytes_iterator tmp(*this);
    operator--();
    return tmp;
  }

  bytes_iterator &operator-=(int i)
  {
    m_data -= i * m_stride;
    return *this;
  }

  bool operator<(const bytes_iterator &rhs) const
  {
    return m_data < rhs.m_data;
  }

  bool operator==(const bytes_iterator &rhs) const
  {
    return m_data == rhs.m_data;
  }

  bool operator!=(const bytes_iterator &rhs) const
  {
    return m_data != rhs.m_data;
  }

  bool operator>(const bytes_iterator &rhs) const
  {
    return m_data > rhs.m_data;
  }

  int operator-(bytes_iterator rhs)
  {
    return (m_data - rhs.m_data) / m_stride;
  }
};

inline bytes_iterator operator+(bytes_iterator lhs, int rhs)
{
  return bytes_iterator(lhs.data() + rhs * lhs.stride(), lhs.size(), lhs.stride());
}

inline bytes_iterator operator-(bytes_iterator lhs, int rhs)
{
  return bytes_iterator(lhs.data() - rhs * lhs.stride(), lhs.size(), lhs.stride());
}

} // namespace dynd

namespace std {

template <>
struct iterator_traits<dynd::bytes_iterator> {
  typedef dynd::std_bytes value_type;
  typedef int difference_type;
  typedef random_access_iterator_tag iterator_category;
};

} // namespace std
