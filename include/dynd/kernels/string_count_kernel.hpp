//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// String count kernel

#pragma once

#include <dynd/string.hpp>

namespace dynd {
namespace nd {

  struct string_count_kernel : base_strided_kernel<string_count_kernel, 2> {
    void single(char *dst, char *const *src)
    {
      intptr_t *d = reinterpret_cast<intptr_t *>(dst);
      const string *const *s = reinterpret_cast<const string *const *>(src);

      *d = dynd::string_count(*(s[0]), *(s[1]));
    }
  };

} // namespace nd
} // namespace dynd
