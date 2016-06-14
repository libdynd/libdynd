//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_apply_callable.hpp>
#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/apply_callable_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {
    namespace detail {

      template <typename func_type, typename ReturnType, typename ArgSequence, int N>
      class apply_callable_callable : public functional::base_apply_callable<func_type> {
        func_type m_func;

      public:
        template <typename... T>
        apply_callable_callable(func_type func, T &&... names)
            : functional::base_apply_callable<func_type>(std::forward<T>(names)...), m_func(func) {}

        ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                          const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
          typedef functional::apply_callable_kernel<func_type, N> kernel_type;

          cg.emplace_back([ func = m_func, kwds = typename kernel_type::kwds_type(nkwd, kwds) ](
              kernel_builder & kb, kernel_request_t kernreq, char *data, const char *dst_arrmeta,
              size_t DYND_UNUSED(narg), const char *const *src_arrmeta) {
            kb.emplace_back<kernel_type>(kernreq, func, typename kernel_type::args_type(data, dst_arrmeta, src_arrmeta),
                                         kwds);
          });

          return this->resolve_return_type(dst_tp, nsrc, src_tp);
        }
      };

    } // namespace dynd::nd::functional::detail

    template <typename FuncType, int N>
    using apply_callable_callable =
        detail::apply_callable_callable<FuncType, typename return_of<FuncType>::type, args_for<FuncType, N>, N>;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
