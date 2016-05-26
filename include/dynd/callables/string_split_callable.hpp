//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/string_split_kernel.hpp>

namespace dynd {
namespace nd {

  class string_split_callable : public base_callable {
  public:
    string_split_callable()
        : base_callable(ndt::make_type<ndt::callable_type>(ndt::make_type<ndt::var_dim_type>(ndt::make_type<string>()),
                                                           {ndt::make_type<string>(), ndt::make_type<string>()})) {}

    ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                      const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                      size_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                      const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
      cg.emplace_back([](kernel_builder &kb, kernel_request_t kernreq, char *DYND_UNUSED(data), const char *dst_arrmeta,
                         size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
        kb.emplace_back<string_split_kernel>(
            kernreq, reinterpret_cast<const ndt::var_dim_type::metadata_type *>(dst_arrmeta)->blockref);
      });

      return dst_tp;
    }
  };

} // namespace dynd::nd
} // namespace dynd
