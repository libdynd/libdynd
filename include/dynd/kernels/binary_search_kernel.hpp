//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/comparison.hpp>
#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  struct binary_search_kernel : base_strided_kernel<binary_search_kernel, 2> {
    const intptr_t src0_size;
    const intptr_t src0_stride;

    binary_search_kernel(intptr_t src0_size, intptr_t src0_stride) : src0_size(src0_size), src0_stride(src0_stride) {}

    void single(char *dst, char *const *src)
    {
      kernel_prefix *child = get_child();

      intptr_t first = 0, last = src0_size;
      while (first < last) {
        intptr_t trial = first + (last - first) / 2;
        char *trial_data = src[0] + trial * src0_stride;

        // In order for the data to always match up with the arrmeta, need to have
        // trial_data first and data second in the comparison operations.
        char *src_try0[2] = {src[1], trial_data};
        char *src_try1[2] = {trial_data, src[1]};
        int child_dst0;
        child->single(reinterpret_cast<char *>(&child_dst0), src_try0);
        if (child_dst0) {
          // value < arr[trial]
          last = trial;
        }
        else {
          int child_dst1;
          child->single(reinterpret_cast<char *>(&child_dst1), src_try1);
          if (child_dst1) {
            // value > arr[trial]
            first = trial + 1;
          }
          else {
            *reinterpret_cast<intptr_t *>(dst) = trial;
            return;
          }
        }
      }

      *reinterpret_cast<intptr_t *>(dst) = -1;
    }
  };

} // namespace dynd::nd
} // namespace dynd
