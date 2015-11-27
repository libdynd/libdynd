//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/bytes.hpp>
#include <dynd/func/comparison.hpp>
#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  struct sort_kernel : base_kernel<sort_kernel, 1> {
    static const size_t data_size = 0;

    const intptr_t src0_size;
    const intptr_t src0_stride;
    const intptr_t src0_element_data_size;

    sort_kernel(intptr_t src0_size, intptr_t src0_stride, size_t src0_element_data_size)
        : src0_size(src0_size), src0_stride(src0_stride), src0_element_data_size(src0_element_data_size)
    {
    }

    ~sort_kernel()
    {
      get_child()->destroy();
    }

    void single(char *DYND_UNUSED(dst), char *const *src)
    {
      ckernel_prefix *child = get_child();
      std::sort(strided_iterator(src[0], src0_element_data_size, src0_stride),
                strided_iterator(src[0] + src0_size * src0_stride, src0_element_data_size, src0_stride),
                [child](char *lhs, char *rhs) {
        bool1 dst;
        char *src[2] = {lhs, rhs};
        child->single(reinterpret_cast<char *>(&dst), src);
        return dst;
      });
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
                                const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      const ndt::type &src0_element_tp = src_tp[0].template extended<ndt::fixed_dim_type>()->get_element_type();

      make(ckb, kernreq, ckb_offset, reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->dim_size,
           reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->stride, src0_element_tp.get_data_size());

      const ndt::type child_src_tp[2] = {src0_element_tp, src0_element_tp};
      const callable &less = nd::less;
      return less.get()->instantiate(less.get()->static_data(), data, ckb, ckb_offset, ndt::type::make<bool1>(), NULL, 2,
                                     child_src_tp, NULL, kernel_request_single, ectx, nkwd, kwds, tp_vars);
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct type::equivalent<nd::sort_kernel> {
    static type make()
    {
      return callable_type::make(type::make<void>(), {type("Fixed * Scalar")});
    }
  };

} // namespace dynd::ndt
} // namespace dynd
