//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ASSIGNMENT_KERNELS_HPP_
#define _DYND__ASSIGNMENT_KERNELS_HPP_

#include <dynd/dtype.hpp>
#include <dynd/dtype_assign.hpp>
#include <dynd/kernels/kernel_instance.hpp>
#include <dynd/eval/eval_context.hpp>

namespace dynd {

/**
 * Returns an assignment function for assigning a built-in
 * dtype value. The errmode must not be `assign_error_default`, which
 * would require an auxdata with a kernel_api to retrieve that default.
 *
 * This returns NULL if there is any problem.
 */
unary_operation_pair_t get_builtin_dtype_assignment_function(type_id_t dst_type_id, type_id_t src_type_id,
                                                                assign_error_mode errmode);

/**
 * A multiple assignment kernel which calls one of the single assignment functions repeatedly.
 * The auxiliary data should be created by calling
 *      make_raw_auxiliary_data(out_auxdata, reinterpret_cast<uintptr_t>(asnFn))
 */
void multiple_assignment_kernel(char *dst, intptr_t dst_stride, const char *src, intptr_t src_stride,
                                    intptr_t count, const AuxDataBase *auxdata);

/**
 * Returns a function for assigning from the source data type
 * to the destination data type. The returned specialization
 * instance contains auxdata and a pointer to a static array
 * of kernel specializations.
 *
 * This function is only for the built-in data types like int,
 * float, complex. A built-in data type can be detected by
 * checking dtype_instance.is_builtin().
 */
void get_builtin_dtype_assignment_kernel(
                    type_id_t dst_type_id, type_id_t src_type_id,
                    assign_error_mode errmode,
                    const eval::eval_context *ectx,
                    kernel_instance<unary_operation_pair_t>& out_kernel);

/**
 * Gets a unary kernel for assigning a pod dtype, i.e. a raw
 * byte-copy. The returned specialization
 * instance contains auxdata and a pointer to a static array
 * of kernel specializations.
 */
void get_pod_dtype_assignment_kernel(
                    intptr_t element_size, intptr_t alignment,
                    kernel_instance<unary_operation_pair_t>& out_kernel);

/**
 * Returns a kernel for assigning from the source data type
 * to the destination data type. The returned specialization
 * instance contains auxdata and a pointer to a static array
 * of kernel specializations.
 */
void get_dtype_assignment_kernel(const dtype& dst_dt, const dtype& src_dt,
                    assign_error_mode errmode,
                    const eval::eval_context *ectx,
                    kernel_instance<unary_operation_pair_t>& out_kernel);

/**
 * Returns a kernel for assigning from the source data to the dest data, with
 * matching source and destination dtypes. The returned specialization
 * instance contains auxdata and a pointer to a static array
 * of kernel specializations.
 */
void get_dtype_assignment_kernel(const dtype& dt,
                    kernel_instance<unary_operation_pair_t>& out_kernel);

} // namespace dynd

#endif // _DYND__ASSIGNMENT_KERNELS_HPP_
