//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd { namespace kernels {

/**
 * Makes a unary reduction ckernel which adds values for the
 * given type id. This is not defined for all type_id values.
 */
intptr_t make_builtin_sum_reduction_ckernel(void *ckb,
                                            intptr_t ckb_offset,
                                            type_id_t tid,
                                            kernel_request_t kernreq);

/**
 * Makes a unary reduction arrfunc for the requested
 * type id.
 */
nd::arrfunc make_builtin_sum_reduction_arrfunc(type_id_t tid);

/**
 * Makes a 1D sum arrfunc.
 * (fixed * <tid>) -> <tid>
 */
nd::arrfunc make_builtin_sum1d_arrfunc(type_id_t tid);

/**
 * Makes a 1D mean arrfunc.
 * (fixed * <tid>) -> <tid>
 */
nd::arrfunc make_builtin_mean1d_arrfunc(type_id_t tid, intptr_t minp);

intptr_t make_strided_reduction_ckernel(void *ckb, intptr_t ckb_offset);

nd::arrfunc make_strided_reduction_arrfunc();

}} // namespace dynd::kernels
