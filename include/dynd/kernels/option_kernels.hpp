//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>

namespace dynd { namespace kernels {

struct fixed_dim_is_avail_ck {
  static intptr_t instantiate(const arrfunc_old_type_data *self,
                              dynd::ckernel_builder *ckb, intptr_t ckb_offset,
                              const ndt::type &dst_tp,
                              const char *dst_arrmeta,
                              const ndt::type *src_tp,
                              const char *const *src_arrmeta,
                              kernel_request_t kernreq,
                              const eval::eval_context *ectx,
                              const nd::array &args,
                              const nd::array &kwds);
};

struct fixed_dim_assign_na_ck {
  static intptr_t instantiate(const arrfunc_old_type_data *self,
                              dynd::ckernel_builder *ckb, intptr_t ckb_offset,
                              const ndt::type &dst_tp,
                              const char *dst_arrmeta,
                              const ndt::type *src_tp,
                              const char *const *src_arrmeta,
                              kernel_request_t kernreq,
                              const eval::eval_context *ectx,
                              const nd::array &args,
                              const nd::array &kwds);
};

/**
 * Returns the nafunc structure for the given builtin type id.
 */
const nd::array &get_option_builtin_nafunc(type_id_t tid);

/**
 * Returns the nafunc structure for the given pointer to builtin type id.
 */
const nd::array &get_option_builtin_pointer_nafunc(type_id_t tid);

}} // namespace dynd::kernels
