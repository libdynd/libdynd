//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_strided_kernel.hpp>
#include <dynd/types/callable_type.hpp>

namespace dynd {
namespace nd {

  template <typename Arg0Type>
  struct max_kernel : base_strided_kernel<max_kernel<Arg0Type>, 1> {
    typedef Arg0Type dst_type;

    void single(char *dst, char *const *src) {
      if (*reinterpret_cast<Arg0Type *>(src[0]) > *reinterpret_cast<dst_type *>(dst)) {
        *reinterpret_cast<dst_type *>(dst) = *reinterpret_cast<Arg0Type *>(src[0]);
      }
    }

    void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count) {
      char *src0 = src[0];
      intptr_t src0_stride = src_stride[0];
      for (size_t i = 0; i < count; ++i) {
        if (*reinterpret_cast<Arg0Type *>(src0) > *reinterpret_cast<dst_type *>(dst)) {
          *reinterpret_cast<dst_type *>(dst) = *reinterpret_cast<Arg0Type *>(src0);
        }
        dst += dst_stride;
        src0 += src0_stride;
      }
    }
  };

  template <>
  struct max_kernel<complex<float>> : base_strided_kernel<max_kernel<complex<float>>, 1> {
    typedef complex<float> dst_type;

    void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src)) {
      throw std::runtime_error("nd::max is not implemented for complex types");
    }
  };

  template <>
  struct max_kernel<complex<double>> : base_strided_kernel<max_kernel<complex<double>>, 1> {
    typedef complex<double> dst_type;

    void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src)) {
      throw std::runtime_error("nd::max is not implemented for complex types");
    }
  };

} // namespace dynd::nd
} // namespace dynd
