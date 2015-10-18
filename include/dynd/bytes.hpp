//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>

namespace dynd {

class DYND_API bytes {
protected:
  char *m_begin;
  char *m_end;

public:
  void assign(char *begin, size_t size)
  {
    m_begin = begin;
    m_end = m_begin + size;
  }

  char *begin()
  {
    return m_begin;
  }

  const char *begin() const
  {
    return m_begin;
  }

  char *end()
  {
    return m_end;
  }

  const char *end() const
  {
    return m_end;
  }
};

} // namespace dynd
