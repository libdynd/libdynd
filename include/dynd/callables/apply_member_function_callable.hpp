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
    public:
      std::pair<T, mem_func_type> m_pair;

      apply_member_function_callable(const ndt::type &tp, T obj, mem_func_type mem_func)
          : base_callable(tp), m_pair(obj, mem_func)
      {
      }

      void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                       const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                       const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const nd::array *kwds,
                       const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        typedef apply_member_function_kernel<T, mem_func_type, N> kernel_type;
        ckb->emplace_back<kernel_type>(kernreq, m_pair.first, dynd::detail::make_value_wrapper(m_pair.second),
                                       typename kernel_type::args_type(src_tp, src_arrmeta, kwds),
                                       typename kernel_type::kwds_type(nkwd, kwds));
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
