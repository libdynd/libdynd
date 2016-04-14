//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/apply_callable_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <typename func_type, int N>
    class apply_callable_callable : public base_callable {
      func_type m_func;

    public:
      template <typename... T>
      apply_callable_callable(func_type func, T &&... names)
          : base_callable(ndt::make_type<typename funcproto_of<func_type>::type>(std::forward<T>(names)...)),
            m_func(func) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                        const ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                        size_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        typedef apply_callable_kernel<func_type, N> kernel_type;

        cg.emplace_back([ func = m_func, kwds = typename kernel_type::kwds_type(nkwd, kwds) ](
            kernel_builder & kb, kernel_request_t kernreq, char *DYND_UNUSED(data),
            const char *DYND_UNUSED(dst_arrmeta), size_t DYND_UNUSED(narg), const char *const *src_arrmeta) {
          kb.emplace_back<kernel_type>(kernreq, func, typename kernel_type::args_type(src_arrmeta, nullptr), kwds);
        });

        return dst_tp;
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
