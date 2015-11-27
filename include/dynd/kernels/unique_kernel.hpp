//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/bytes.hpp>
#include <dynd/kernels/base_kernel.hpp>

namespace dynd {
namespace nd {

  struct unique_kernel : base_kernel<unique_kernel> {
    static const size_t data_size = 0;

    const intptr_t src0_size;
    const intptr_t src0_stride;
    const intptr_t src0_element_data_size;

    unique_kernel(intptr_t src0_size, intptr_t src0_stride, size_t src0_element_data_size)
        : src0_size(src0_size), src0_stride(src0_stride), src0_element_data_size(src0_element_data_size)
    {
    }

    ~unique_kernel() { get_child()->destroy(); }

    void single(array *DYND_UNUSED(dst), array *const *src)
    {
      ckernel_prefix *child = get_child();
      size_t new_size =
          (std::unique(strided_iterator(src[0]->data(), src0_element_data_size, src0_stride),
                       strided_iterator(src[0]->data() + src0_size * src0_stride, src0_element_data_size, src0_stride),
                       [child](char *lhs, char *rhs) {
                         bool1 dst;
                         char *src[2] = {lhs, rhs};
                         child->single(reinterpret_cast<char *>(&dst), src);
                         return dst;
                       }) -
           src[0]->data()) /
          src0_stride;

      src[0]->get()->tp =
          ndt::make_fixed_dim(new_size, src[0]->get()->tp.extended<ndt::fixed_dim_type>()->get_element_type());
      reinterpret_cast<size_stride_t *>(src[0]->get()->metadata())->dim_size = new_size;
    }

    static intptr_t instantiate(char *DYND_UNUSED(static_data), char *data, void *ckb, intptr_t ckb_offset,
                                const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                                kernel_request_t kernreq, const eval::eval_context *ectx, intptr_t nkwd,
                                const nd::array *kwds, const std::map<std::string, ndt::type> &tp_vars)
    {
      const ndt::type &src0_element_tp = src_tp[0].extended<ndt::fixed_dim_type>()->get_element_type();
      make(ckb, kernreq, ckb_offset, reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->dim_size,
           reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->stride, src0_element_tp.get_data_size());

      const callable &equal = nd::equal;
      const ndt::type equal_src_tp[2] = {src0_element_tp, src0_element_tp};
      return equal.get()->instantiate(equal.get()->static_data(), data, ckb, ckb_offset, ndt::type::make<bool1>(), NULL,
                                      2, equal_src_tp, NULL, kernel_request_single, ectx, nkwd, kwds, tp_vars);
    }
  };

} // namespace dynd::nd

namespace ndt {

  template <>
  struct type::equivalent<nd::unique_kernel> {
    static type make() { return callable_type::make(type::make<void>(), {type("Fixed * Scalar")}); }
  };

} // namespace dynd::ndt
} // namespace dynd
