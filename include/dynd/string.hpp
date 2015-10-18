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

  string(char *data, size_t size) : bytes(data, size)
  {
  }
};

bool operator<(const string &lhs, const string &rhs);
bool operator<=(const string &lhs, const string &rhs);
bool operator==(const string &lhs, const string &rhs);
bool operator!=(const string &lhs, const string &rhs);
bool operator>=(const string &lhs, const string &rhs);
bool operator>(const string &lhs, const string &rhs);

} // namespace dynd
