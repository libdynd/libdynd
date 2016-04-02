//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    // Right associative, evaluate the reduction from right to left:
    //    dst_(0) = a[n-1]
    //    dst_(i+1) = dst_(i) <OP> a[n-1-(i+1)]
    struct DYND_API left_compound_kernel : base_strided_kernel<left_compound_kernel, 1> {
      ~left_compound_kernel() { get_child()->destroy(); }

      void single(char *dst, char *const *src)
      {
        kernel_prefix *child = get_child();
        kernel_single_t single = child->get_function<kernel_single_t>();
        char *src_binary[2] = {dst, src[0]};
        single(child, dst, src_binary);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        kernel_prefix *child = get_child();
        kernel_strided_t childop = child->get_function<kernel_strided_t>();
        char *src_binary[2] = {dst, src[0]};
        const intptr_t src_binary_stride[2] = {dst_stride, src_stride[0]};
        childop(child, dst, dst_stride, src_binary, src_binary_stride, count);
      }
    };

    // Left associative, evaluate the reduction from left to right:
    //    dst_(0) = a[0]
    //    dst_(i+1) = a[i+1] <OP> dst_(i)
    struct DYND_API right_compound_kernel : base_strided_kernel<right_compound_kernel, 1> {
      ~right_compound_kernel() { get_child()->destroy(); }

      void single(char *dst, char *const *src)
      {
        kernel_prefix *child = get_child();
        kernel_single_t childop = child->get_function<kernel_single_t>();
        char *src_binary[2] = {src[0], dst};
        childop(child, dst, src_binary);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
      {
        kernel_prefix *child = get_child();
        kernel_strided_t childop = child->get_function<kernel_strided_t>();
        char *src_binary[2] = {src[0], dst};
        const intptr_t src_binary_stride[2] = {src_stride[0], dst_stride};
        childop(child, dst, dst_stride, src_binary, src_binary_stride, count);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
