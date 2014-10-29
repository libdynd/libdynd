//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/time_type.hpp>

namespace dynd {

/**
 * Makes a kernel which converts strings to times.
 */
size_t make_string_to_time_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_time_tp,
    const ndt::type &src_string_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx);

/**
 * Makes a kernel which converts times to strings.
 */
size_t make_time_to_string_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_string_tp,
    const char *dst_arrmeta, const ndt::type &src_time_tp,
    kernel_request_t kernreq, const eval::eval_context *ectx);

} // namespace dynd
