//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/expr_kernels.hpp>

namespace dynd {
namespace kernels {
  /**
   * Generic expr kernel + destructor for a strided dimension with
   * a fixed number of src operands.
   * This requires that the child kernel be created with the
   * kernel_request_strided type of kernel.
   */
  template <int N>
  struct elwise : expr_ck<elwise<N>, kernel_request_cuda_host_device, N> {
    typedef elwise self_type;

    intptr_t size;
    intptr_t dst_stride, src_stride[N];

    DYND_CUDA_HOST_DEVICE elwise(intptr_t size, intptr_t dst_stride,
                                 const intptr_t *src_stride)
        : size(size), dst_stride(dst_stride)
    {
      memcpy(this->src_stride, src_stride, sizeof(this->src_stride));
    }

    DYND_CUDA_HOST_DEVICE void single(char *dst, char **src)
    {
      ckernel_prefix *child = this->get_child_ckernel();
      expr_strided_t opchild = child->get_function<expr_strided_t>();
      opchild(dst, this->dst_stride, src, this->src_stride, this->size, child);
    }

    DYND_CUDA_HOST_DEVICE void strided(char *dst, intptr_t dst_stride,
                                       char **src, const intptr_t *src_stride,
                                       size_t count)
    {
      ckernel_prefix *child = this->get_child_ckernel();
      expr_strided_t opchild = child->get_function<expr_strided_t>();
      intptr_t inner_size = this->size, inner_dst_stride = this->dst_stride;
      const intptr_t *inner_src_stride = this->src_stride;
      char *src_loop[N];
      memcpy(src_loop, src, sizeof(src_loop));
      for (size_t i = 0; i != count; ++i) {
        opchild(dst, inner_dst_stride, src_loop, inner_src_stride, inner_size,
                child);
        dst += dst_stride;
        for (int j = 0; j != N; ++j) {
          src_loop[j] += src_stride[j];
        }
      }
    }

    DYND_CUDA_HOST_DEVICE static void destruct(ckernel_prefix *self)
    {
      self->destroy_child_ckernel(sizeof(self_type));
    }
  };
} // namespace dynd
} // namespace kernels