//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/comparison.hpp>
#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/binary_search_kernel.hpp>

namespace dynd {
namespace nd {

  class binary_search_callable : public base_callable {
  public:
    binary_search_callable()
        : base_callable(ndt::callable_type::make(ndt::make_type<intptr_t>(),
                                                 {ndt::type("Fixed * Scalar"), ndt::type("Scalar")})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(call_node *&node, char *data, kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                     const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                     const nd::array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &tp_vars) {
      ckb->emplace_back<binary_search_kernel>(
          kernreq, reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->dim_size,
          reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->stride);
      node = next(node);

      const char *n_arrmeta = src_arrmeta[0];
      ndt::type element_tp = src_tp[0].at_single(0, &n_arrmeta);

      ndt::type child_src_tp[2] = {element_tp, element_tp};
      const char *child_src_arrmeta[2] = {n_arrmeta, n_arrmeta};

      total_order->instantiate(node, data, ckb, ndt::make_type<int>(), NULL, 2, child_src_tp, child_src_arrmeta,
                               kernreq | kernel_request_data_only, 0, NULL, tp_vars);
    }
  };

} // namespace dynd::nd
} // namespace dynd
