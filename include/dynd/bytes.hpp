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

} // namespace dynd
