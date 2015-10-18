//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/bytes.hpp>

namespace dynd {

class DYND_API string : public bytes {
};

bool operator<(const string &lhs, const string &rhs);
bool operator<=(const string &lhs, const string &rhs);
bool operator==(const string &lhs, const string &rhs);
bool operator!=(const string &lhs, const string &rhs);
bool operator>=(const string &lhs, const string &rhs);
bool operator>(const string &lhs, const string &rhs);

} // namespace dynd
