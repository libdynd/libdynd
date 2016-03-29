//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_instantiable_callable.hpp>
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
            m_func(func)
      {
        m_new_style = true;
      }

      void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                       const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                       const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                       const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        typedef apply_callable_kernel<func_type, N> kernel_type;
        ckb->emplace_back<kernel_type>(kernreq, m_func, typename kernel_type::args_type(src_tp, src_arrmeta, kwds),
                                       typename kernel_type::kwds_type(nkwd, kwds));
      }

      void new_instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                           const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                           const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const array *kwds)
      {
        typedef apply_callable_kernel<func_type, N> kernel_type;
        ckb->emplace_back<kernel_type>(kernreq, m_func, typename kernel_type::args_type(src_tp, src_arrmeta, kwds),
                                       typename kernel_type::kwds_type(nkwd, kwds));
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd