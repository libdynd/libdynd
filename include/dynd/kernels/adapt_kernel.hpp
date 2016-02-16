//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/types/adapt_type.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    struct adapt_kernel : base_kernel<adapt_kernel> {
      struct static_data_type {
        ndt::type value_tp;
        callable forward;
      };

      const ndt::type &value_tp;
      const callable &forward;

      adapt_kernel(const ndt::type &value_tp, const callable &forward) : value_tp(value_tp), forward(forward) {}

      void call(array *dst, const array *src)
      {
        *dst = src[0].replace_dtype(ndt::make_type<ndt::adapt_type>(value_tp, src[0].get_dtype(), forward, callable()));
      }

      static void resolve_dst_type(char *static_data, char *DYND_UNUSED(data), ndt::type &dst_tp,
                                   intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                   intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                                   const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        dst_tp = reinterpret_cast<static_data_type *>(static_data)->value_tp;
      }

      static void instantiate(char *static_data, char *DYND_UNUSED(data), kernel_builder *ckb,
                              const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                              intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                              const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                              intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                              const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        ckb->emplace_back<adapt_kernel>(kernreq, reinterpret_cast<static_data_type *>(static_data)->value_tp,
                                        reinterpret_cast<static_data_type *>(static_data)->forward);
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
