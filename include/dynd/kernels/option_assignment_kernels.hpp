//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__OPTION_ASSIGNMENT_KERNELS_HPP_
#define _DYND__OPTION_ASSIGNMENT_KERNELS_HPP_

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd { namespace kernels {

/**
 * Makes a ckernel for assignments containing an option type.
 */
size_t make_option_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp,
    const char *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx);
}} // namespace dynd::kernels

#endif // _DYND__OPTION_ASSIGNMENT_KERNELS_HPP_
