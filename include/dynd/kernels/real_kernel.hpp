//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename Arg0Type>
  struct real_kernel : base_strided_kernel<real_kernel<Arg0Type>, 1> {
    void single(char *dst, char *const *src) {
      *reinterpret_cast<typename Arg0Type::value_type *>(dst) = reinterpret_cast<Arg0Type *>(src[0])->real();
    }
  };

} // namespace dynd::nd
} // namespace dynd
