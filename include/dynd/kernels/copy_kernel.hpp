//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_virtual_kernel.hpp>

namespace dynd {
namespace nd {

  struct DYND_API copy_ck : base_virtual_kernel<copy_ck> {
    static void resolve_dst_type(char *static_data, char *data, ndt::type &dst_tp, intptr_t nsrc,
                                 const ndt::type *src_tp, intptr_t nkwd, const array *kwds,
                                 const std::map<std::string, ndt::type> &tp_vars);

    static intptr_t instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
                                const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
                                const char *const *src_arrmeta, kernel_request_t kernreq,
                                const eval::eval_context *ectx, intptr_t nkwd, const nd::array *kwds,
                                const std::map<std::string, ndt::type> &tp_vars);
  };

} // namespace dynd::nd
} // namespace dynd
