//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename Arg0Type, typename Arg1Type>
  struct all_equal_kernel : base_strided_kernel<all_equal_kernel<Arg0Type, Arg1Type>, 2> {
    typedef typename std::common_type<Arg0Type, Arg1Type>::type T;

    void single(char *dst, char *const *src) {
      *reinterpret_cast<bool *>(dst) &= static_cast<T>(*reinterpret_cast<Arg0Type *>(src[0])) ==
                                        static_cast<T>(*reinterpret_cast<Arg1Type *>(src[1]));
    }
  };

  template <>
  struct all_equal_kernel<ndt::tuple_type, ndt::tuple_type>
      : base_strided_kernel<all_equal_kernel<ndt::tuple_type, ndt::tuple_type>, 2> {
    void single(char *DYND_UNUSED(dst), char *const *DYND_UNUSED(src)) {
      // ...
      // ...
    }
  };

} // namespace dynd::nd
} // namespace dynd
