//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    struct constant_kernel : base_strided_kernel<constant_kernel, 0> {
      char *data;

      constant_kernel(char *data) : data(data) {}

      ~constant_kernel() { get_child()->destroy(); }

      void single(char *dst, char *const *DYND_UNUSED(src)) { get_child()->single(dst, &data); }

      void strided(char *dst, intptr_t dst_stride, char *const *DYND_UNUSED(src),
                   const intptr_t *DYND_UNUSED(src_stride), size_t count)
      {
        static std::intptr_t data_stride[1] = {0};

        get_child()->strided(dst, dst_stride, &data, data_stride, count);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
