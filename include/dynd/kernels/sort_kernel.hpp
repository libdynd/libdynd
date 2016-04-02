//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/bytes.hpp>
#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {

  struct sort_kernel : base_strided_kernel<sort_kernel, 1> {
    const intptr_t src0_size;
    const intptr_t src0_stride;
    const intptr_t src0_element_data_size;

    sort_kernel(intptr_t src0_size, intptr_t src0_stride, size_t src0_element_data_size)
        : src0_size(src0_size), src0_stride(src0_stride), src0_element_data_size(src0_element_data_size)
    {
    }

    ~sort_kernel() { get_child()->destroy(); }

    void single(char *DYND_UNUSED(dst), char *const *src)
    {
      kernel_prefix *child = get_child();
      std::sort(strided_iterator(src[0], src0_element_data_size, src0_stride),
                strided_iterator(src[0] + src0_size * src0_stride, src0_element_data_size, src0_stride),
                [child](char *lhs, char *rhs) {
                  bool1 dst;
                  char *src[2] = {lhs, rhs};
                  child->single(reinterpret_cast<char *>(&dst), src);
                  return dst;
                });
    }
  };

} // namespace dynd::nd
} // namespace dynd
