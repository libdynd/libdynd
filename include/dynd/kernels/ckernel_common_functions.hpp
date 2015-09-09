//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/types/type_id.hpp>

namespace dynd {

namespace kernels {

/**
 * A function to destroy a ckernel which doesn't have
 * any extra data, and has a single child kernel.
 */
//void destroy_trivial_parent_ckernel(ckernel_prefix *ckp);

/**
 * An expr single ckernel function which adapts a child
 * unary single ckernel.
 */
DYND_API void unary_as_expr_adapter_single_ckernel(
                char *dst, const char * const *src,
                ckernel_prefix *ckp);

/**
 * An expr strided ckernel function which adapts a child
 * unary strided ckernel.
 */
DYND_API void unary_as_expr_adapter_strided_ckernel(
                char *dst, intptr_t dst_stride,
                const char * const *src, const intptr_t *src_stride,
                size_t count, ckernel_prefix *ckp);

/**
 * A unary single ckernel function which adapts a child
 * expr single ckernel. The child expr single ckernel
 * must itself be unary as well.
 */
DYND_API void expr_as_unary_adapter_single_ckernel(
                char *dst, const char *src,
                ckernel_prefix *ckp);

/**
 * A unary strided ckernel function which adapts a child
 * expr strided ckernel. The child expr single ckernel
 * must itself be unary as well.
 */
DYND_API void expr_as_unary_adapter_strided_ckernel(
                char *dst, intptr_t dst_stride,
                const char *src, intptr_t src_stride,
                size_t count, ckernel_prefix *ckp);

/**
 * Adds an adapter ckernel which wraps a child expr ckernel
 * as a unary ckernel. The child expr single ckernel
 * must itself be unary as well.
 *
 * \returns  The ckb_offset where the child ckernel should be placed.
 */
DYND_API intptr_t wrap_expr_as_unary_ckernel(void *ckb,
                                    intptr_t ckb_offset,
                                    kernel_request_t kernreq);

} // namespace kernels

} // namespace dynd
