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

    static void instantiate(char *DYND_UNUSED(static_data), char *data, kernel_builder *ckb,
                            const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                            intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                            kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd), const nd::array *DYND_UNUSED(kwds),
                            const std::map<std::string, ndt::type> &tp_vars)
    {
      ckb->emplace_back<binary_search_kernel>(
          kernreq, reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->dim_size,
          reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->stride);

      const char *n_arrmeta = src_arrmeta[0];
      ndt::type element_tp = src_tp[0].at_single(0, &n_arrmeta);

      ndt::type child_src_tp[2] = {element_tp, element_tp};
      const char *child_src_arrmeta[2] = {n_arrmeta, n_arrmeta};

      total_order::get().get()->instantiate(total_order::get().get()->static_data(), data, ckb, ndt::make_type<int>(),
                                            NULL, 2, child_src_tp, child_src_arrmeta,
                                            kernreq | kernel_request_data_only, 0, NULL, tp_vars);
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct traits<nd::binary_search_kernel> {
    static type equivalent()
    {
      return callable_type::make(make_type<intptr_t>(), {type("Fixed * Scalar"), type("Scalar")});
    }
  };

} // namespace dynd::ndt
} // namespace dynd
