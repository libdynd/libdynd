//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {

  typedef intptr_t stride_t;

  template <type_id_t RetElementID>
  struct range_kernel : base_strided_kernel<range_kernel<RetElementID>, 0> {
    typedef typename type_of<RetElementID>::type type;

    type start;
    type stop;
    type step;
    size_t size;
    stride_t stride;

    range_kernel(type start, type stop, type step, size_t size, stride_t stride)
        : start(start), stop(stop), step(step), size(size), stride(stride) {}

    void single(char *ret, char *const *DYND_UNUSED(args)) {
      for (size_t i = 0; i < size; ++i) {
        *reinterpret_cast<type *>(ret) = start + i * step;
        ret += stride;
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
