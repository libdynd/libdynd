//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

// Implement a string concatenation kernel

#pragma once

#include <dynd/string.hpp>

namespace dynd {
namespace nd {

  struct string_concatenation_kernel : base_strided_kernel<string_concatenation_kernel, 2> {
    void single(char *dst, char *const *src)
    {
      dynd::string_concat(2, *reinterpret_cast<string *>(dst), reinterpret_cast<const string *const *>(src));
    }
  };

} // namespace nd
} // namespace dynd
