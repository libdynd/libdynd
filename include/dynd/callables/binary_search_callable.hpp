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
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &tp_vars) {
      cg.emplace_back([](kernel_builder &kb, kernel_request_t kernreq, const char *DYND_UNUSED(dst_arrmeta),
                         size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
        kb.emplace_back<binary_search_kernel>(
            kernreq, reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->dim_size,
            reinterpret_cast<const fixed_dim_type_arrmeta *>(src_arrmeta[0])->stride);

        const char *child_src_arrmeta[2] = {src_arrmeta[0], src_arrmeta[0]};
        kb.instantiate(kernreq | kernel_request_data_only, nullptr, 2, child_src_arrmeta);
      });

      ndt::type element_tp = src_tp[0].at_single(0, nullptr);
      ndt::type child_src_tp[2] = {element_tp, element_tp};

      total_order->resolve(this, nullptr, cg, ndt::make_type<int>(), 2, child_src_tp, 0, NULL, tp_vars);

      return dst_tp;
    }
  };

} // namespace dynd::nd
} // namespace dynd
