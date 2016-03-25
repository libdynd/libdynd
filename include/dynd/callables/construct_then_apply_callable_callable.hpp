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

    template <typename func_type, typename... K>
    class construct_then_apply_callable_callable : public base_callable {
    public:
      construct_then_apply_callable_callable(const ndt::type &tp) : base_callable(tp) {}

      void instantiate(char *DYND_UNUSED(data), kernel_builder *ckb, const ndt::type &DYND_UNUSED(dst_tp),
                       const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
                       const char *const *src_arrmeta, kernel_request_t kernreq, intptr_t nkwd, const array *kwds,
                       const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        typedef construct_then_apply_callable_kernel<func_type, K...> kernel_type;
        ckb->emplace_back<kernel_type>(kernreq, typename kernel_type::args_type(src_tp, src_arrmeta, kwds),
                                       typename kernel_type::kwds_type(nkwd, kwds));
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
