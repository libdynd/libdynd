//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/apply_function_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <typename func_type, func_type func, int N = arity_of<func_type>::value>
    class apply_function_callable : public base_callable {
    public:
      template <typename... T>
      apply_function_callable(T &&... names)
          : base_callable(ndt::make_type<typename funcproto_of<func_type>::type>(std::forward<T>(names)...)) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                        size_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        typedef apply_function_kernel<func_type, func, N> kernel_type;

        cg.emplace_back([kwds = typename kernel_type::kwds_type(nkwd, kwds)](
            kernel_builder & kb, kernel_request_t kernreq, char *DYND_UNUSED(data), const char *DYND_UNUSED(dst_arrmeta),
            size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
          kb.emplace_back<kernel_type>(kernreq, typename kernel_type::args_type(src_arrmeta, nullptr), kwds);
        });

        return dst_tp;
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
