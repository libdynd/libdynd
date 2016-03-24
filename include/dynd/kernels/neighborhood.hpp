//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <algorithm>

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/substitute_shape.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <int N>
    struct neighborhood_kernel : base_strided_kernel<neighborhood_kernel<N>, N> {
      intptr_t dst_stride;
      intptr_t src0_offset;
      intptr_t src0_stride;
      intptr_t offset;
      intptr_t counts[3];
      std::shared_ptr<bool> out_of_bounds;
      intptr_t boundary_child_offset;

      neighborhood_kernel(intptr_t dst_stride, intptr_t src0_size, intptr_t src0_stride, intptr_t size, intptr_t offset,
                          const std::shared_ptr<bool> &out_of_bounds)
          : dst_stride(dst_stride), src0_offset(offset * src0_stride), src0_stride(src0_stride), offset(offset),
            out_of_bounds(out_of_bounds)
      {
        counts[0] = std::min((intptr_t)0, src0_size + offset);
        counts[1] = std::min(src0_size + offset, src0_size - size + 1);
        counts[2] = src0_size + offset;

        *out_of_bounds = false;
      }

      void single(char *dst, char *const *src)
      {
        kernel_prefix *child = this->get_child();
        kernel_prefix *boundary_child = this->get_child(boundary_child_offset);

        char *src0 = src[0] + src0_offset;

        intptr_t i = offset;
        bool old_out_of_bounds = *out_of_bounds;

        *out_of_bounds = true;
        while (i < counts[0]) {
          boundary_child->single(dst, &src0);
          ++i;
          dst += dst_stride;
          src0 += src0_stride;
        };

        *out_of_bounds = old_out_of_bounds;
        while (i < counts[1]) {
          if (*out_of_bounds) {
            boundary_child->single(dst, &src0);
          }
          else {
            child->single(dst, &src0);
          }
          ++i;
          dst += dst_stride;
          src0 += src0_stride;
        }

        *out_of_bounds = true;
        while (i < counts[2]) {
          boundary_child->single(dst, &src0);
          ++i;
          dst += dst_stride;
          src0 += src0_stride;
        }

        *out_of_bounds = old_out_of_bounds;
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
