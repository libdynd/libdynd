//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/fixed_string_type.hpp>

namespace dynd {
namespace nd {

  template <typename Arg0Type, typename Arg1Type>
  struct total_order_kernel;

  template <>
  struct total_order_kernel<bool, bool> : base_strided_kernel<total_order_kernel<bool, bool>, 2> {
    void single(char *dst, char *const *src) {
      *reinterpret_cast<int *>(dst) =
          static_cast<int>(*reinterpret_cast<bool1 *>(src[0])) < static_cast<int>(*reinterpret_cast<bool1 *>(src[1]));
    }
  };

  template <>
  struct total_order_kernel<int32_t, int32_t> : base_strided_kernel<total_order_kernel<int32_t, int32_t>, 2> {
    void single(char *dst, char *const *src) {
      *reinterpret_cast<int *>(dst) = *reinterpret_cast<int *>(src[0]) < *reinterpret_cast<int *>(src[1]);
    }
  };

  template <>
  struct total_order_kernel<string, string> : base_strided_kernel<total_order_kernel<string, string>, 2> {
    void single(char *dst, char *const *src) {
      *reinterpret_cast<int *>(dst) = std::lexicographical_compare(
          reinterpret_cast<string *>(src[0])->begin(), reinterpret_cast<string *>(src[0])->end(),
          reinterpret_cast<string *>(src[1])->begin(), reinterpret_cast<string *>(src[1])->end());
    }
  };

} // namespace dynd::nd
} // namespace dynd
