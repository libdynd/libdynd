//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {

/**
 * Makes a kernel which does an assignment when
 * at least one of dst_dt and src_dt is an
 * expr_kind type.
 */
DYND_API void make_expression_assignment_kernel(nd::kernel_builder *ckb, const ndt::type &dst_tp,
                                                const char *dst_arrmeta, const ndt::type &src_tp,
                                                const char *src_arrmeta, kernel_request_t kernreq,
                                                const eval::eval_context *ectx);

} // namespace dynd
