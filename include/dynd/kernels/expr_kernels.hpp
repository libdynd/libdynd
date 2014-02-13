//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__EXPR_KERNELS_HPP_
#define _DYND__EXPR_KERNELS_HPP_

#include <dynd/kernels/expr_kernel_generator.hpp>

namespace dynd {

/**
 * Evaluates any expression types in the array of
 * source types, passing the result non-expression
 * types on to the handler to build the rest of the
 * kernel.
 */
size_t make_expression_type_expr_kernel(ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_tp, const char *dst_metadata,
                size_t src_count, const ndt::type *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const expr_kernel_generator *handler);

} // namespace dynd

#endif // _DYND__EXPR_KERNELS_HPP_
