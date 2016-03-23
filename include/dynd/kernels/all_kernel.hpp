//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {

  struct all_kernel : base_strided_kernel<all_kernel, 1> {
    void single(char *dst, char *const *src)
    {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<bool1 *>(dst) && *reinterpret_cast<bool1 *>(src[0]);
    }

    void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
    {
      char *src0 = src[0];
      intptr_t src0_stride = src_stride[0];
      for (size_t i = 0; i < count; ++i) {
        *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<bool1 *>(dst) && *reinterpret_cast<bool1 *>(src0);
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
