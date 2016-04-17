//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/where_kernel.hpp>

namespace dynd {
namespace nd {

  class where_callable : public base_callable {
    callable m_child;

  public:
    where_callable(const callable &child)
        : base_callable(ndt::callable_type::make(
              ndt::make_type<ndt::var_dim_type>(ndt::make_type<ndt::fixed_dim_type>(1, ndt::make_type<index_t>())),
              child->get_argument_types())),
          m_child(child) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back([](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data), const char *dst_arrmeta,
                         size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
        kb.emplace_back<where_kernel>(
            kernreq, reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta)->blockref);
      });

      return dst_tp;
    }
  };

} // namespace dynd::nd
} // namespace dynd
