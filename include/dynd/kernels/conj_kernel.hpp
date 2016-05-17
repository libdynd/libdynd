//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename Arg0Type>
  struct conj_kernel : base_strided_kernel<conj_kernel<Arg0Type>, 1> {
    void single(char *dst, char *const *src) {
      *reinterpret_cast<Arg0Type *>(dst) = dynd::conj(*reinterpret_cast<Arg0Type *>(src[0]));
    }
  };

} // namespace dynd::nd
} // namespace dynd
