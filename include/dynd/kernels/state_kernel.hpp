//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>
#include <dynd/types/iteration_type.hpp>

namespace dynd {
namespace nd {

  template <size_t NArg>
  struct state_kernel : base_strided_kernel<state_kernel<NArg>, NArg> {
    state st;

    void single(char *dst, char *const *src) {
      char *child_src[NArg + 1];
      for (size_t i = 0; i < NArg; ++i) {
        child_src[i] = src[i];
      }
      child_src[NArg] = reinterpret_cast<char *>(&st);

      this->get_child()->single(dst, child_src);
    }
  };

} // namespace dynd::nd
} // namespace dynd
