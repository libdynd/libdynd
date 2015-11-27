//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/comparison.hpp>
#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  struct binary_search_kernel : base_kernel<binary_search_kernel, 2> {
    static const size_t data_size = 0;

    const intptr_t src0_size;
    const intptr_t src0_stride;

    binary_search_kernel(intptr_t src0_size, intptr_t src0_stride) : src0_size(src0_size), src0_stride(src0_stride) {}

    void single(char *dst, char *const *src)
    {
      ckernel_prefix *child = get_child();

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

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t DYND_UNUSED(nkwd),
                                const nd::array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &tp_vars)
    {
      make(ckb, kernreq, ckb_offset, reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->dim_size,
           reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->stride);

      const char *n_arrmeta = src_arrmeta[0];
      ndt::type element_tp = src_tp[0].at_single(0, &n_arrmeta);

      ndt::type child_src_tp[2] = {element_tp, element_tp};
      const char *child_src_arrmeta[2] = {n_arrmeta, n_arrmeta};

      return total_order::get().get()->instantiate(total_order::get().get()->static_data(), data, ckb, ckb_offset,
                                                   ndt::type::make<int>(), NULL, 2, child_src_tp, child_src_arrmeta,
                                                   kernreq, ectx, 0, NULL, tp_vars);
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct type::equivalent<nd::binary_search_kernel> {
    static type make() { return callable_type::make(type::make<intptr_t>(), {type("Fixed * Scalar"), type("Scalar")}); }
  };

} // namespace dynd::ndt
} // namespace dynd
