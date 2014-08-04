//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__EXPRESSION_ASSIGNMENT_KERNELS_HPP_
#define _DYND__EXPRESSION_ASSIGNMENT_KERNELS_HPP_

#include <dynd/kernels/assignment_kernels.hpp>

namespace dynd {

/**
 * Makes a kernel which does an assignment when
 * at least one of dst_dt and src_dt is an
 * expr_kind type.
 */
size_t make_expression_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__EXPRESSION_ASSIGNMENT_KERNELS_HPP_
