//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__MAKE_LIFTED_CKERNEL_HPP_
#define _DYND__MAKE_LIFTED_CKERNEL_HPP_

#include <dynd/config.hpp>
#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {

/**
 * Lifts the provided reduction ckernel, broadcasting against the output
 *
 * This lifts a unary_operation ckernel, which accumulates an individual value
 * or a strided run of values with the possibility of a 0-stride in the output.
 *
 * \param elwise_reduction  The arrfunc being lifted
 * \param dst_initialization  A arrfunc for initializing the
 *                            accumulator values from the source data.
 *                            If this is NULL, an assignment
 *                            kernel is used here.
 * \param ckb  The ckernel_builder into which to place the ckernel.
 * \param ckb_offset  Where within the ckernel_builder to place the ckernel.
 * \param dst_tp  The destination type to lift to.
 * \param dst_arrmeta  The destination arrmeta to lift to.
 * \param src_tp  The source type to lift to.
 * \param src_arrmeta  The source arrmeta to lift to.
 * \param reduction_ndim  The number of dimensions being reduced.
 * \param reduction_dimflags  Boolean flags indicating which dimensions to
 *                            reduce. This can typically be derived from an
 *                            "axis=" parameter.
 * \param associative  Whether we can assume the reduction kernel is
 *                     associative.
 * \param commutative  Whether we can assume the reduction kernel is
 *                     commutative.
 * \param right_associative  If true, the reduction is to be evaluated right to
 *                           left instead of left to right.
 * \param reduction_identity  Either a NULL array if there is no identity, or
 *                            a value that the output can be initialized to at
 *                            the start.
 * \param kernreq  Either dynd::kernel_request_single or
 *                 dynd::kernel_request_strided,
 *                 as required by the caller.
 * \param ectx  The evaluation context to use.
 */
size_t make_lifted_reduction_ckernel(
    const arrfunc_type_data *elwise_reduction,
    const arrfunc_type_data *dst_initialization, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type &src_tp, const char *src_arrmeta, intptr_t reduction_ndim,
    const bool *reduction_dimflags, bool associative, bool commutative,
    bool right_associative, const nd::array &reduction_identity,
    dynd::kernel_request_t kernreq,
    const eval::eval_context *ectx = &eval::default_eval_context);

} // namespace dynd

#endif // _DYND__MAKE_LIFTED_CKERNEL_HPP_
 
