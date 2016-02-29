//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// String replace kernel

#pragma once

#include <dynd/string.hpp>

namespace dynd {
namespace nd {

  struct string_replace_kernel : base_strided_kernel<string_replace_kernel, 3> {
    void single(char *dst, char *const *src)
    {
      string *d = reinterpret_cast<string *>(dst);
      const string *const *s = reinterpret_cast<const string *const *>(src);

      dynd::string_replace(*d, *s[0], *s[1], *s[2]);
    }
  };

} // namespace nd

namespace ndt {

  template <>
  struct traits<dynd::nd::string_replace_kernel> {
    static type equivalent()
    {
      return callable_type::make(type(string_id), {type(string_id), type(string_id), type(string_id)});
    }
  };

} // namespace ndt

} // namespace dynd
