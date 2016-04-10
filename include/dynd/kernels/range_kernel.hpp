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
    stride_t stride;

    range_kernel(type start, type stop, type step) : start(start), stop(stop), step(step), stride(sizeof(type)) {}

    void single(char *ret, char *const *DYND_UNUSED(args)) {
      for (type i = start; i < stop; i += step) {
        *reinterpret_cast<type *>(ret) = i;
        ret += stride;
      }
    }
  };

} // namespace dynd::nd
} // namespace dynd
