//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd { namespace kernels {

/**
 * Makes a ckernel for assignments containing an option type.
 */
size_t make_option_assignment_kernel(
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp,
    const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx);
}} // namespace dynd::kernels
