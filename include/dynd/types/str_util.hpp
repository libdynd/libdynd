//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

namespace dynd {

inline bool is_simple_identifier_name(const std::string &s)
{
  if (s.empty()) {
    return false;
  }
  else {
    auto c = s.begin();
    if (!(('a' <= *c && *c <= 'z') || ('A' <= *c && *c <= 'Z') || *c == '_')) {
      return false;
    }
    for (++c; c != s.end(); ++c) {
      if (!(('0' <= *c && *c <= '9') || ('a' <= *c && *c <= 'z') || ('A' <= *c && *c <= 'Z') || *c == '_')) {
        return false;
      }
    }
    return true;
  }
}

} // namespace dynd
