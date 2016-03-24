//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/bytes.hpp>
#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  struct unique_kernel : base_kernel<unique_kernel> {
    const intptr_t src0_size;
    const intptr_t src0_stride;
    const intptr_t src0_element_data_size;

    unique_kernel(intptr_t src0_size, intptr_t src0_stride, size_t src0_element_data_size)
        : src0_size(src0_size), src0_stride(src0_stride), src0_element_data_size(src0_element_data_size)
    {
    }

    ~unique_kernel() { get_child()->destroy(); }

    void call(array *DYND_UNUSED(dst), const array *src)
    {
      kernel_prefix *child = get_child();
      size_t new_size =
          (std::unique(strided_iterator(src[0].data(), src0_element_data_size, src0_stride),
                       strided_iterator(src[0].data() + src0_size * src0_stride, src0_element_data_size, src0_stride),
                       [child](char *lhs, char *rhs) {
                         bool1 dst;
                         char *src[2] = {lhs, rhs};
                         child->single(reinterpret_cast<char *>(&dst), src);
                         return dst;
                       }) -
           src[0].data()) /
          src0_stride;

      src[0]->tp = ndt::make_fixed_dim(new_size, src[0]->tp.extended<ndt::fixed_dim_type>()->get_element_type());
      reinterpret_cast<size_stride_t *>(src[0]->metadata())->dim_size = new_size;
    }
  };

} // namespace dynd::nd
} // namespace dynd
