//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>

namespace dynd {

struct DYND_API string {
  char *begin;
  char *end;
};

bool operator<(const string &lhs, const string &rhs);
bool operator<=(const string &lhs, const string &rhs);
bool operator==(const string &lhs, const string &rhs);
bool operator!=(const string &lhs, const string &rhs);
bool operator>=(const string &lhs, const string &rhs);
bool operator>(const string &lhs, const string &rhs);

} // namespace dynd
