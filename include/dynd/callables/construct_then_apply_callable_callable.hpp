//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/construct_then_apply_callable_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <typename func_type, typename... KwdTypes>
    class construct_then_apply_callable_callable : public base_callable {
    public:
      template <typename... T>
      construct_then_apply_callable_callable(T &&... names)
          : base_callable(
                ndt::make_type<typename funcproto_of<func_type, KwdTypes...>::type>(std::forward<T>(names)...)) {}

      void resolve(call_graph &cg) { cg.emplace_back(this); }

      void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                       const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                       const ndt::type *DYND_UNUSED(src_tp), const char *const *src_arrmeta, kernel_request_t kernreq,
                       intptr_t nkwd, const array *kwds, const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        typedef construct_then_apply_callable_kernel<func_type, KwdTypes...> kernel_type;
        ckb->emplace_back<kernel_type>(kernreq, typename kernel_type::args_type(src_arrmeta, kwds),
                                       typename kernel_type::kwds_type(nkwd, kwds));
      }

      void new_instantiate(call_frame *DYND_UNUSED(frame), kernel_builder &ckb, kernel_request_t kernreq,
                           const char *DYND_UNUSED(dst_arrmeta), const char *const *src_arrmeta, size_t nkwd,
                           const array *kwds) {
        typedef construct_then_apply_callable_kernel<func_type, KwdTypes...> kernel_type;
        ckb.emplace_back<kernel_type>(kernreq, typename kernel_type::args_type(src_arrmeta, kwds),
                                      typename kernel_type::kwds_type(nkwd, kwds));
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
