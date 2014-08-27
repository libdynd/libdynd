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
intptr_t make_builtin_sum_reduction_ckernel(ckernel_builder *ckb,
                                            intptr_t ckb_offset,
                                            type_id_t tid,
                                            kernel_request_t kernreq);

/**
 * Makes a unary reduction arrfunc for the requested
 * type id.
 */
void make_builtin_sum_reduction_arrfunc(arrfunc_type_data *out_af,
                                        type_id_t tid);

/**
 * Makes a unary reduction arrfunc for the requested
 * type id.
 */
inline nd::arrfunc make_builtin_sum_reduction_arrfunc(type_id_t tid)
{
    nd::array af = nd::empty(ndt::make_arrfunc());
    make_builtin_sum_reduction_arrfunc(
        reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()),
        tid);
    af.flag_as_immutable();
    return af;
}

/**
 * Makes a 1D sum arrfunc.
 * (strided * <tid>) -> <tid>
 */
nd::arrfunc make_builtin_sum1d_arrfunc(type_id_t tid);

/**
 * Makes a 1D mean arrfunc.
 * (strided * <tid>) -> <tid>
 */
nd::arrfunc make_builtin_mean1d_arrfunc(type_id_t tid, intptr_t minp);

}} // namespace dynd::kernels

#endif // _DYND__REDUCTION_KERNELS_HPP_
