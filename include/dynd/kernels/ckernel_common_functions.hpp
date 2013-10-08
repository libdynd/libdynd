//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CKERNEL_COMMON_FUNCTIONS_HPP_
#define _DYND__CKERNEL_COMMON_FUNCTIONS_HPP_

#include <dynd/config.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/types/type_id.hpp>

namespace dynd {

namespace kernels {

/**
 * A function to destroy a ckernel which doesn't have
 * any extra data, and has a single child kernel.
 */
void destroy_trivial_parent_ckernel(ckernel_prefix *ckp);

/**
 * An expr single ckernel function which adapts a child
 * unary single ckernel.
 */
void unary_as_expr_adapter_single_ckernel(
                char *dst, const char * const *src,
                ckernel_prefix *ckp);

/**
 * An expr strided ckernel function which adapts a child
 * unary strided ckernel.
 */
void unary_as_expr_adapter_strided_ckernel(
                char *dst, intptr_t dst_stride,
                const char * const *src, const intptr_t *src_stride,
                size_t count, ckernel_prefix *ckp);

/**
 * Adds an adapter ckernel which wraps a child unary ckernel
 * as an expr ckernel.
 *
 * \returns  The ckb_offset where the child ckernel should be placed.
 */
intptr_t wrap_unary_as_expr_ckernel(
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                kernel_request_t kerntype);

} // namespace kernels

} // namespace dynd

#endif // _DYND__CKERNEL_COMMON_FUNCTIONS_HPP_
