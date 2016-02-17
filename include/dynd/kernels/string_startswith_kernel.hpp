//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// String startswith kernel

#pragma once

#include <dynd/string.hpp>

namespace dynd {
namespace nd {

  struct string_startswith_kernel : base_strided_kernel<string_startswith_kernel, 2> {
    void single(char *dst, char *const *src)
    {
      bool1 *d = reinterpret_cast<bool1 *>(dst);
      const string *const *s = reinterpret_cast<const string *const *>(src);

      *d = dynd::string_startswith(*(s[0]), *(s[1]));
    }
  };

} // namespace nd
} // namespace dynd
