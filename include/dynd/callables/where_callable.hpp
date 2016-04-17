//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/where_kernel.hpp>

namespace dynd {
namespace ndt {

  inline type make_where(const nd::callable &child) {
    std::vector<type> arg_tp = child->get_argument_types();
    arg_tp.push_back(make_type<nd::state>());

    return ndt::callable_type::make(
        ndt::make_type<ndt::var_dim_type>(ndt::make_type<ndt::fixed_dim_type>(1, ndt::make_type<nd::index_t>())),
        arg_tp);
  }

} // namespace dynd::ndt

namespace nd {

  class where_callable : public base_callable {
    callable m_child;

  public:
    where_callable(const callable &child) : base_callable(ndt::make_where(child)), m_child(child) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                      const std::map<std::string, ndt::type> &tp_vars) {
      cg.emplace_back([](kernel_builder &kb, kernel_request_t kernreq, char *data, const char *dst_arrmeta, size_t nsrc,
                         const char *const *src_arrmeta) {
        kb.emplace_back<where_kernel>(
            kernreq, data, reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta)->stride,
            reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta)->blockref);

        kb(kernel_request_single, nullptr, nullptr, nsrc - 1, src_arrmeta);
      });

      m_child->resolve(this, nullptr, cg, ndt::make_type<bool>(), nsrc - 1, src_tp, nkwd, kwds, tp_vars);

      return dst_tp;
    }
  };

} // namespace dynd::nd
} // namespace dynd
