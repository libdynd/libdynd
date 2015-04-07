//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_virtual_kernel.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    struct outer_ck : base_virtual_kernel<outer_ck> {
      static intptr_t instantiate(
          const arrfunc_type_data *self, const arrfunc_type *self_tp,
          char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
          const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
          const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
          const eval::eval_context *ectx, const dynd::nd::array &kwds,
          const std::map<dynd::nd::string, ndt::type> &tp_vars);

      static void
      resolve_dst_type(const arrfunc_type_data *self,
                       const arrfunc_type *self_tp, char *data,
                       ndt::type &dst_tp, intptr_t nsrc,
                       const ndt::type *src_tp, const dynd::nd::array &kwds,
                       const std::map<dynd::nd::string, ndt::type> &tp_vars);
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd