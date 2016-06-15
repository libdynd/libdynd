//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_apply_callable.hpp>
#include <dynd/kernels/apply_member_function_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {
    namespace detail {

      template <typename T, typename mem_func_type, typename ReturnType, typename ArgSequence, typename KwdSequence,
                int N>
      class apply_member_function_callable : public base_apply_callable<ReturnType, ArgSequence, KwdSequence> {
        T m_obj;
        mem_func_type m_mem_func;

      public:
        template <typename... S>
        apply_member_function_callable(T obj, mem_func_type mem_func, S &&... names)
            : base_apply_callable<ReturnType, ArgSequence, KwdSequence>(std::forward<S>(names)...), m_obj(obj),
              m_mem_func(mem_func) {}

        ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg,
                          const ndt::type &dst_tp, size_t nsrc, const ndt::type *src_tp, size_t nkwd, const array *kwds,
                          const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
          typedef functional::apply_member_function_kernel<T, mem_func_type, N> kernel_type;

          cg.emplace_back([ obj = m_obj, mem_func = m_mem_func, kwds = typename kernel_type::kwds_type(nkwd, kwds) ](
              kernel_builder & kb, kernel_request_t kernreq, char *data, const char *dst_arrmeta,
              size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
            kb.emplace_back<kernel_type>(kernreq, obj, mem_func,
                                         typename kernel_type::args_type(data, dst_arrmeta, src_arrmeta), kwds);
          });

          return this->resolve_return_type(dst_tp, nsrc, src_tp);
        }
      };

    } // namespace dynd::nd::functional::detail

    template <typename T, typename mem_func_type, int N>
    using apply_member_function_callable =
        detail::apply_member_function_callable<T, mem_func_type, typename return_of<mem_func_type>::type,
                                               args_for<mem_func_type, N>, as_apply_kwd_sequence<mem_func_type, N>, N>;

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
