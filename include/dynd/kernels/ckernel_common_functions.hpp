//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
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
 * A unary single ckernel function which adapts a child
 * expr single ckernel. The child expr single ckernel
 * must itself be unary as well.
 */
void expr_as_unary_adapter_single_ckernel(
                char *dst, const char *src,
                ckernel_prefix *ckp);

/**
 * A unary strided ckernel function which adapts a child
 * expr strided ckernel. The child expr single ckernel
 * must itself be unary as well.
 */
void expr_as_unary_adapter_strided_ckernel(
                char *dst, intptr_t dst_stride,
                const char *src, intptr_t src_stride,
                size_t count, ckernel_prefix *ckp);

/**
 * Makes a ckernel that ignores the src values, and writes
 * constant values to the output.
 *
 */
size_t make_constant_value_assignment_ckernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const nd::array &constant,
    kernel_request_t kernreq, const eval::eval_context *ectx);

/**
 * Adds an adapter ckernel which wraps a child expr ckernel
 * as a unary ckernel. The child expr single ckernel
 * must itself be unary as well.
 *
 * \returns  The ckb_offset where the child ckernel should be placed.
 */
intptr_t wrap_expr_as_unary_ckernel(dynd::ckernel_builder *ckb,
                                    intptr_t ckb_offset,
                                    kernel_request_t kernreq);

/**
 * Adds an adapter ckernel which wraps a child binary expr ckernel
 * as a unary reduction ckernel. The three types of the binary
 * expr kernel must all be equal.
 *
 * \param ckb  The ckernel_builder into which the kernel adapter is placed.
 * \param ckb_offset  The offset within the ckernel_builder at which to place the adapter.
 * \param right_associative  If true, the reduction is to be evaluated right to left,
 *                           (x0 * (x1 * (x2 * x3))), if false, the reduction is to be
 *                           evaluted left to right (((x0 * x1) * x2) * x3).
 * \param kernreq  The type of kernel to produce (single or strided).
 *
 * \returns  The ckb_offset where the child ckernel should be placed.
 */
intptr_t wrap_binary_as_unary_reduction_ckernel(
                dynd::ckernel_builder *ckb, intptr_t ckb_offset,
                bool right_associative,
                kernel_request_t kernreq);

} // namespace kernels

} // namespace dynd

#endif // _DYND__CKERNEL_COMMON_FUNCTIONS_HPP_
