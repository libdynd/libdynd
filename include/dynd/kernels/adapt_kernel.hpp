//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

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

      void single(array *dst, array *const *src)
      {
        *dst = src[0]->replace_dtype(
            ndt::make_type<ndt::new_adapt_type>(value_tp, src[0]->get_dtype(), forward, callable()));
      }

      static void resolve_dst_type(char *static_data, char *DYND_UNUSED(data), ndt::type &dst_tp,
                                   intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                   intptr_t DYND_UNUSED(nkwd), const array *DYND_UNUSED(kwds),
                                   const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        dst_tp = reinterpret_cast<static_data_type *>(static_data)->value_tp;
      }

      static intptr_t instantiate(char *static_data, char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                                  const ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                                  intptr_t DYND_UNUSED(nsrc), const ndt::type *DYND_UNUSED(src_tp),
                                  const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
                                  const eval::eval_context *DYND_UNUSED(ectx), intptr_t DYND_UNUSED(nkwd),
                                  const array *DYND_UNUSED(kwds),
                                  const std::map<std::string, ndt::type> &DYND_UNUSED(tp_vars))
      {
        make(ckb, kernreq, ckb_offset, reinterpret_cast<static_data_type *>(static_data)->value_tp,
             reinterpret_cast<static_data_type *>(static_data)->forward);
        return ckb_offset;
      }
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
