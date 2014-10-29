//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {

/**
 * Instantiate an arrfunc, adding buffers for any inputs where the types
 * don't match.
 */
size_t make_buffered_ckernel(const arrfunc_type_data *af,
                             dynd::ckernel_builder *ckb, intptr_t ckb_offset,
                             const ndt::type &dst_tp, const char *dst_arrmeta,
                             intptr_t nsrc, const ndt::type *src_tp,
                             const ndt::type *src_tp_for_af,
                             const char *const *src_arrmeta,
                             kernel_request_t kernreq,
                             const eval::eval_context *ectx);

} // namespace dynd
