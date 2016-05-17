//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {

  typedef intptr_t stride_t;

  template <typename ReturnElementType>
  struct range_kernel : base_strided_kernel<range_kernel<ReturnElementType>, 0> {
    ReturnElementType start;
    ReturnElementType stop;
    ReturnElementType step;
    size_t size;
    stride_t stride;

    range_kernel(ReturnElementType start, ReturnElementType stop, ReturnElementType step, size_t size, stride_t stride)
        : start(start), stop(stop), step(step), size(size), stride(stride) {}

    void single(char *ret, char *const *DYND_UNUSED(args)) {
      for (size_t i = 0; i < size; ++i) {
        *reinterpret_cast<ReturnElementType *>(ret) = start + static_cast<ReturnElementType>(i) * step;
        ret += stride;
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
