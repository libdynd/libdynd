//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_callable.hpp>
#include <dynd/kernels/apply_member_function_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    template <typename T, typename mem_func_type, int N>
    class apply_member_function_callable : public base_callable {
      T m_obj;
      mem_func_type m_mem_func;

    public:
      template <typename... S>
      apply_member_function_callable(T obj, mem_func_type mem_func, S &&... names)
          : base_callable(ndt::make_type<mem_func_type>(std::forward<S>(names)...)), m_obj(obj), m_mem_func(mem_func) {}

      ndt::type resolve(base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), call_graph &cg, const ndt::type &dst_tp,
                        size_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                        const array *DYND_UNUSED(kwds), const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        cg.emplace_back(this);
        return dst_tp;
      }

      void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                       const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
                       const ndt::type *DYND_UNUSED(src_tp), const char *const *src_arrmeta, kernel_request_t kernreq,
                       intptr_t nkwd, const nd::array *kwds,
                       const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars)) {
        typedef apply_member_function_kernel<T, mem_func_type, N> kernel_type;
        ckb->emplace_back<kernel_type>(kernreq, m_obj, m_mem_func, typename kernel_type::args_type(src_arrmeta, kwds),
                                       typename kernel_type::kwds_type(nkwd, kwds));
      }

      void new_instantiate(call_frame *DYND_UNUSED(frame), kernel_builder &ckb, kernel_request_t kernreq,
                           const char *DYND_UNUSED(dst_arrmeta), const char *const *src_arrmeta, size_t nkwd,
                           const array *kwds) {
        typedef apply_member_function_kernel<T, mem_func_type, N> kernel_type;
        ckb.emplace_back<kernel_type>(kernreq, m_obj, m_mem_func, typename kernel_type::args_type(src_arrmeta, kwds),
                                      typename kernel_type::kwds_type(nkwd, kwds));
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
