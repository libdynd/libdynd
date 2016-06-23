//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {
  namespace limits {

    template <typename T>
    struct min_kernel : base_strided_kernel<min_kernel<T>, 0> {
      void single(char *dst, char *const *DYND_UNUSED(src)) {
        *reinterpret_cast<T *>(dst) = std::numeric_limits<T>::lowest();
      }
    };

  } // namespace dynd::nd::limits
} // namespace dynd::nd
} // namespace dynd
