//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_apply_callable.hpp>
#include <dynd/kernels/apply_function_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {
    namespace detail {

      template <typename FuncType, FuncType func, typename ReturnType, typename ArgSequence, typename KwdSequence,
                size_t NArg = arity_of<FuncType>::value>
      class apply_function_callable : public base_apply_callable<ReturnType, ArgSequence, KwdSequence> {
      public:
        using base_apply_callable<ReturnType, ArgSequence, KwdSequence>::base_apply_callable;

        ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                          const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
          typedef nd::functional::apply_function_kernel<FuncType, func, NArg> kernel_type;

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

    template <typename FuncType, FuncType Func, size_t NArg = arity_of<FuncType>::value>
    using apply_function_callable =
        detail::apply_function_callable<FuncType, Func, typename return_of<FuncType>::type, args_for<FuncType, NArg>,
                                        as_apply_kwd_sequence<FuncType, NArg>, NArg>;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
