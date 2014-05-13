//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__REDUCTION_KERNELS_HPP_
#define _DYND__REDUCTION_KERNELS_HPP_

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd { namespace kernels {

/**
 * Makes a unary reduction ckernel which adds values for the
 * given type id. This is not defined for all type_id values.
 */
intptr_t make_builtin_sum_reduction_ckernel(
                ckernel_builder *out_ckb, intptr_t ckb_offset,
                type_id_t tid,
                kernel_request_t kerntype);

/**
 * Makes a unary reduction ckernel_deferred for the requested
 * type id.
 */
void make_builtin_sum_reduction_ckernel_deferred(
                ckernel_deferred *out_ckd,
                type_id_t tid);

/**
 * Makes a 1D sum ckernel_deferred.
 * (strided * <tid>) -> <tid>
 */
nd::array make_builtin_sum1d_ckernel_deferred(type_id_t tid);

/**
 * Makes a 1D mean ckernel_deferred.
 * (strided * <tid>) -> <tid>
 */
nd::array make_builtin_mean1d_ckernel_deferred(type_id_t tid, intptr_t minp);

}} // namespace dynd::kernels

#endif // _DYND__REDUCTION_KERNELS_HPP_
