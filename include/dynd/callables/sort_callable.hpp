//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/comparison.hpp>
#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/sort_kernel.hpp>

namespace dynd {
namespace nd {

  class sort_callable : public base_callable {
  public:
    sort_callable() : base_callable(ndt::callable_type::make(ndt::make_type<void>(), {ndt::type("Fixed * Scalar")})) {}

    void resolve(call_graph &cg) { cg.emplace_back(this); }

    void instantiate(char *data, kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                     const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                     const std::map<std::string, ndt::type> &tp_vars) {
      const ndt::type &src0_element_tp = src_tp[0].template extended<ndt::fixed_dim_type>()->get_element_type();

      ckb->emplace_back<sort_kernel>(
          kernreq, reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->dim_size,
          reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->stride, src0_element_tp.get_data_size());

      const ndt::type child_src_tp[2] = {src0_element_tp, src0_element_tp};
      less->instantiate(data, ckb, ndt::make_type<bool1>(), NULL, 2, child_src_tp, NULL, kernel_request_single, nkwd,
                        kwds, tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd
