//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {

  template <size_t NArg>
  struct outer_kernel : base_strided_kernel<outer_kernel<NArg>, NArg> {
    size_t dst_size;
    intptr_t dst_stride;
    intptr_t src_stride[NArg];

    outer_kernel(size_t i, const char *dst_metadata, const char *const *src_metadata)
        : dst_size(reinterpret_cast<const size_stride_t *>(dst_metadata)->dim_size),
          dst_stride(reinterpret_cast<const size_stride_t *>(dst_metadata)->stride) {
      for (size_t j = 0; j < i; ++j) {
        src_stride[j] = 0;
      }
      src_stride[i] = reinterpret_cast<const size_stride_t *>(src_metadata[i])->stride;
      for (size_t j = i + 1; j < NArg; ++j) {
        src_stride[j] = 0;
      }
    }

    ~outer_kernel() { this->get_child()->destroy(); }

    void single(char *dst, char *const *src) {
      kernel_prefix *child = this->get_child();

      child->strided(dst, dst_stride, src, src_stride, dst_size);
    }
  };

  template <>
  struct outer_kernel<0> : base_strided_kernel<outer_kernel<0>, 0> {
    size_t dst_size;
    intptr_t dst_stride;

    outer_kernel(size_t DYND_UNUSED(i), const char *dst_metadata, const char *const *DYND_UNUSED(src_metadata))
        : dst_size(reinterpret_cast<const size_stride_t *>(dst_metadata)->dim_size),
          dst_stride(reinterpret_cast<const size_stride_t *>(dst_metadata)->stride) {}

    ~outer_kernel() { this->get_child()->destroy(); }

    void single(char *dst, char *const *DYND_UNUSED(src)) {
      kernel_prefix *child = this->get_child();

      child->strided(dst, dst_stride, nullptr, nullptr, dst_size);
    }
  };

} // namespace dynd::nd
} // namespace dynd
