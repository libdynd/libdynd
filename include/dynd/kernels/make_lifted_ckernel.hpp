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
 * Lifts the provided ckernel, broadcasting it as necessary to execute
 * across the additional dimensions in the ``lifted_types`` array.
 *
 * This version is for 'expr' ckernels.
 *
 * \param elwise_handler  The arrfunc being lifted
 * \param ckb  The ckernel_builder into which to place the ckernel.
 * \param ckb_offset  Where within the ckernel_builder to place the ckernel.
 * \param dst_ndim  The number of destination dimensions to lift.
 * \param dst_tp  The destination type to lift to.
 * \param dst_arrmeta  The destination arrmeta to lift to.
 * \param src_ndim  The number of dimensions to lift for each source type.
 * \param src_tp  The source types to lift to.
 * \param src_arrmeta  The source arrmetas to lift to.
 * \param kernreq  Either dynd::kernel_request_single or dynd::kernel_request_strided,
 *                  as required by the caller.
 * \param ectx  The evaluation context.
 */
size_t make_lifted_expr_ckernel(
    const arrfunc_type_data *elwise_handler, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, intptr_t dst_ndim, const ndt::type &dst_tp,
    const char *dst_arrmeta, const intptr_t *src_ndim, const ndt::type *src_tp,
    const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
    const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__MAKE_LIFTED_CKERNEL_HPP_
