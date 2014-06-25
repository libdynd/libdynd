//
// Copyright (C) 2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__VAR_DIM_ASSIGNMENT_KERNELS_HPP_
#define _DYND__VAR_DIM_ASSIGNMENT_KERNELS_HPP_

#include <dynd/type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

namespace dynd {

/**
 * Makes a kernel which broadcasts the input to a var dim.
 */
size_t make_broadcast_to_var_dim_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_var_dim_dt,
    const char *dst_arrmeta, const ndt::type &src_tp, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx);

/**
 * Makes a kernel which assigns var dims.
 */
size_t make_var_dim_assignment_kernel(ckernel_builder *ckb, intptr_t ckb_offset,
                                      const ndt::type &dst_var_dim_dt,
                                      const char *dst_arrmeta,
                                      const ndt::type &src_var_dim_dt,
                                      const char *src_arrmeta,
                                      kernel_request_t kernreq,
                                      const eval::eval_context *ectx);

/**
 * Makes a kernel which assigns strided dims to var dims
 */
size_t make_strided_to_var_dim_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset, const ndt::type &dst_var_dim_dt,
    const char *dst_arrmeta, intptr_t src_dim_size, intptr_t src_stride,
    const ndt::type &src_el_tp, const char *src_el_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx);

/**
 * Makes a kernel which assigns var dims to strided dims
 */
size_t make_var_to_strided_dim_assignment_kernel(
    ckernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type &dst_strided_dim_dt, const char *dst_arrmeta,
    const ndt::type &src_var_dim_dt, const char *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__VAR_DIM_ASSIGNMENT_KERNELS_HPP_
