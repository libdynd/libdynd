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

  bytes(bytes &&) = delete;

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
    }

    std::copy_n(data, size, m_data);
    m_size = size;

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

} // namespace dynd
