//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/total_order_kernel.hpp>

namespace dynd {
namespace nd {

  template <type_id_t Arg0ID, type_id_t Arg1ID>
  class total_order_callable : public default_instantiable_callable<total_order_kernel<Arg0ID, Arg1ID>> {
  public:
    total_order_callable()
        : default_instantiable_callable<total_order_kernel<Arg0ID, Arg1ID>>(
              ndt::callable_type::make(ndt::make_type<int>(), {ndt::type(Arg0ID), ndt::type(Arg1ID)})) {}
  };

  template <>
  class total_order_callable<fixed_string_id, fixed_string_id> : public base_callable {
  public:
    total_order_callable()
        : base_callable(ndt::callable_type::make(ndt::make_type<int>(),
                                                 {ndt::type(fixed_string_id), ndt::type(fixed_string_id)})) {}

    const ndt::type &resolve(call_graph &cg, const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                             const ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                             const array *DYND_UNUSED(kwds),
                             const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                     const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq, intptr_t DYND_UNUSED(nkwd),
                     const nd::array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      ckb->emplace_back<total_order_kernel<fixed_string_id, fixed_string_id>>(
          kernreq, src_tp[0].extended<ndt::fixed_string_type>()->get_size());
    }
  };

} // namespace dynd::nd
} // namespace dynd
