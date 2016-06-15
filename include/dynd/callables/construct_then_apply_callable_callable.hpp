//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_apply_callable.hpp>
#include <dynd/kernels/construct_then_apply_callable_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {
    namespace detail {

      template <typename CallableType, typename ReturnType, typename ArgSequence, typename KwdSequence,
                typename... KwdTypes>
      class construct_then_apply_callable_callable : public base_apply_callable<ReturnType, ArgSequence, KwdSequence> {
      public:
        using base_apply_callable<ReturnType, ArgSequence, KwdSequence>::base_apply_callable;

        ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                          const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
          typedef functional::construct_then_apply_callable_kernel<CallableType, KwdTypes...> kernel_type;

          cg.emplace_back([kwds = typename kernel_type::kwds_type(nkwd, kwds)](
              kernel_builder & kb, kernel_request_t kernreq, char *data, const char *dst_arrmeta,
              size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
            kb.emplace_back<kernel_type>(kernreq, typename kernel_type::args_type(data, dst_arrmeta, src_arrmeta),
                                         kwds);
          });

          return this->resolve_return_type(dst_tp, nsrc, src_tp);
        }
      };

    } // namespace dynd::nd::functional::detail

    template <typename CallableType, typename... KwdTypes>
    using construct_then_apply_callable_callable =
        detail::construct_then_apply_callable_callable<CallableType, typename return_of<CallableType>::type,
                                                       args_for<CallableType, arity_of<CallableType>::value>,
                                                       type_sequence<KwdTypes...>, KwdTypes...>;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
