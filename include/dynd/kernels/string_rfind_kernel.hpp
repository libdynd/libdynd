//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// String rfind kernel

#pragma once

#include <dynd/string.hpp>

namespace dynd {
namespace nd {

  struct string_rfind_kernel : base_strided_kernel<string_rfind_kernel, 2> {
    void single(char *dst, char *const *src)
    {
      intptr_t *d = reinterpret_cast<intptr_t *>(dst);
      const string *const *s = reinterpret_cast<const string *const *>(src);

      *d = dynd::string_rfind(*(s[0]), *(s[1]));
    }
  };

} // namespace nd

namespace ndt {

  template <>
  struct traits<nd::string_rfind_kernel> {
    static type equivalent()
    {
      return callable_type::make(ndt::make_type<intptr_t>(), {type(string_id), type(string_id)});
    }
  };

} // namespace ndt

} // namespace dynd
