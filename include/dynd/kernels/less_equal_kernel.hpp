//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_strided_kernel.hpp>

namespace dynd {
namespace nd {

  template <typename Arg0Type, typename Arg1Type>
  struct less_equal_kernel : base_strided_kernel<less_equal_kernel<Arg0Type, Arg1Type>, 2> {
    typedef typename std::common_type<Arg0Type, Arg1Type>::type T;

    void single(char *dst, char *const *src) {
      *reinterpret_cast<bool1 *>(dst) = static_cast<T>(*reinterpret_cast<Arg0Type *>(src[0])) <=
                                        static_cast<T>(*reinterpret_cast<Arg1Type *>(src[1]));
    }
  };

  template <typename Arg0Type>
  struct less_equal_kernel<Arg0Type, Arg0Type> : base_strided_kernel<less_equal_kernel<Arg0Type, Arg0Type>, 2> {
    void single(char *dst, char *const *src) {
      *reinterpret_cast<bool1 *>(dst) = *reinterpret_cast<Arg0Type *>(src[0]) <= *reinterpret_cast<Arg0Type *>(src[1]);
    }
  };

} // namespace dynd::nd
} // namespace dynd
