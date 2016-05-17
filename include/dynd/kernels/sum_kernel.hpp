//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename Arg0Type>
  struct sum_kernel : base_strided_kernel<sum_kernel<Arg0Type>, 1> {
    typedef Arg0Type dst_type;

    void single(char *dst, char *const *src) {
      *reinterpret_cast<dst_type *>(dst) = *reinterpret_cast<dst_type *>(dst) + *reinterpret_cast<Arg0Type *>(src[0]);
    }

    void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count) {
      char *src0 = src[0];
      intptr_t src0_stride = src_stride[0];
      for (size_t i = 0; i < count; ++i) {
        *reinterpret_cast<dst_type *>(dst) = *reinterpret_cast<dst_type *>(dst) + *reinterpret_cast<Arg0Type *>(src0);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
