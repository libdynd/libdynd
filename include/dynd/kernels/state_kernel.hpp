//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>
#include <dynd/types/state_type.hpp>

namespace dynd {
namespace nd {

  template <size_t NArg>
  struct state_kernel : base_strided_kernel<state_kernel<NArg>, NArg> {
    size_t i;
    state st;

    state_kernel(size_t i) : i(i) {}

    ~state_kernel() {
      delete[] st.index;
      this->get_child()->destroy();
    }

    void single(char *dst, char *const *src) {
      char *child_src[NArg + 1];
      for (size_t j = 0; j < i; ++j) {
        child_src[j] = src[j];
      }
      child_src[i] = reinterpret_cast<char *>(&st);
      for (size_t j = i; j < NArg; ++j) {
        child_src[j + 1] = src[j];
      }

      this->get_child()->single(dst, child_src);
    }

    void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t end) {
      char *child_src[NArg + 1];
      intptr_t child_src_stride[NArg + 1];
      for (size_t j = 0; j < i; ++j) {
        child_src[j] = src[j];
        child_src_stride[j] = src_stride[j];
      }
      child_src[i] = reinterpret_cast<char *>(&st);
      child_src_stride[i] = 0;
      for (size_t j = i; j < NArg; ++j) {
        child_src[j + 1] = src[j];
        child_src_stride[j + 1] = src_stride[j];
      }

      this->get_child()->strided(dst, dst_stride, child_src, child_src_stride, end);
    }
  };

} // namespace dynd::nd
} // namespace dynd
