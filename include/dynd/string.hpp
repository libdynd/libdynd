//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/bytes.hpp>

namespace dynd {

class DYND_API string : public bytes {
public:
  string()
  {
  }

  string(const char *data, size_t size) : bytes(data, size)
  {
  }

  bool operator<(const string &rhs) const
  {
    return std::lexicographical_compare(begin(), end(), rhs.begin(), rhs.end());
  }

  bool operator<=(const string &rhs) const
  {
    return !std::lexicographical_compare(rhs.begin(), rhs.end(), begin(), end());
  }

  bool operator>=(const string &rhs) const
  {
    return !std::lexicographical_compare(begin(), end(), rhs.begin(), rhs.end());
  }

  bool operator>(const string &rhs) const
  {
    return std::lexicographical_compare(rhs.begin(), rhs.end(), begin(), end());
  }
};

} // namespace dynd
