//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/callable_type.hpp>

namespace dynd {
namespace nd {

  template <typename Arg0Type, typename Arg1Type>
  struct not_equal_kernel : base_strided_kernel<not_equal_kernel<Arg0Type, Arg1Type>, 2> {
    typedef typename std::common_type<Arg0Type, Arg1Type>::type T;

    void single(char *dst, char *const *src) {
      *reinterpret_cast<bool1 *>(dst) = static_cast<T>(*reinterpret_cast<Arg0Type *>(src[0])) !=
                                        static_cast<T>(*reinterpret_cast<Arg1Type *>(src[1]));
    }
  };

  template <typename Arg0Type>
  struct not_equal_kernel<Arg0Type, Arg0Type> : base_strided_kernel<not_equal_kernel<Arg0Type, Arg0Type>, 2> {
    void single(char *res, char *const *args) {
      *reinterpret_cast<bool1 *>(res) =
          *reinterpret_cast<Arg0Type *>(args[0]) != *reinterpret_cast<Arg0Type *>(args[1]);
    }
  };

  template <>
  struct not_equal_kernel<ndt::tuple_type, ndt::tuple_type>
      : base_strided_kernel<not_equal_kernel<ndt::tuple_type, ndt::tuple_type>, 2> {
    size_t field_count;
    const size_t *src0_data_offsets, *src1_data_offsets;
    // After this are field_count sorting_less kernel offsets, for
    // src0.field_i <op> src1.field_i
    // with each 0 <= i < field_count

    not_equal_kernel(size_t field_count, const size_t *src0_data_offsets, const size_t *src1_data_offsets)
        : field_count(field_count), src0_data_offsets(src0_data_offsets), src1_data_offsets(src1_data_offsets) {}

    ~not_equal_kernel() {
      size_t *kernel_offsets = reinterpret_cast<size_t *>(this + 1);
      for (size_t i = 0; i != field_count; ++i) {
        get_child(kernel_offsets[i])->destroy();
      }
    }

    void single(char *dst, char *const *src) {
      const size_t *kernel_offsets = reinterpret_cast<const size_t *>(this + 1);
      char *child_src[2];
      for (size_t i = 0; i != field_count; ++i) {
        kernel_prefix *echild = reinterpret_cast<kernel_prefix *>(reinterpret_cast<char *>(this) + kernel_offsets[i]);
        kernel_single_t opchild = echild->get_function<kernel_single_t>();
        // if (src0.field_i < src1.field_i) return true
        child_src[0] = src[0] + src0_data_offsets[i];
        child_src[1] = src[1] + src1_data_offsets[i];
        bool1 child_dst;
        opchild(echild, reinterpret_cast<char *>(&child_dst), child_src);
        if (child_dst) {
          *reinterpret_cast<bool1 *>(dst) = true;
          return;
        }
      }
      *reinterpret_cast<bool1 *>(dst) = false;
    }
  };

} // namespace dynd::nd
} // namespace dynd
