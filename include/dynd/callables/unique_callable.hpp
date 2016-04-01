//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/unique_kernel.hpp>

namespace dynd {
namespace nd {

  class unique_callable : public base_callable {
  public:
    unique_callable()
        : base_callable(ndt::callable_type::make(ndt::make_type<void>(), {ndt::type("Fixed * Scalar")})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(call_node *DYND_UNUSED(node), char *data, kernel_builder *ckb,
                     const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                     intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
                     kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      const ndt::type &src0_element_tp = src_tp[0].extended<ndt::fixed_dim_type>()->get_element_type();
      ckb->emplace_back<unique_kernel>(
          kernreq, reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->dim_size,
          reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->stride, src0_element_tp.get_data_size());

      const ndt::type equal_src_tp[2] = {src0_element_tp, src0_element_tp};
      equal->instantiate(nullptr, data, ckb, ndt::make_type<bool1>(), NULL, 2, equal_src_tp, NULL,
                         kernel_request_single, nkwd, kwds, tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd
