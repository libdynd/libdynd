//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__ASSIGNMENT_KERNELS_HPP_
#define _DND__ASSIGNMENT_KERNELS_HPP_

#include <dnd/dtype.hpp>
#include <dnd/dtype_assign.hpp>

namespace dnd {

/**
 * A function prototype for functions which assign a single value.
 */
typedef void (*assignment_function_t)(void *dst, const void *src);

/**
 * Returns an assignment function for assigning a single built-in
 * dtype value.
 */
assignment_function_t get_builtin_dtype_assignment_function(type_id_t dst_type_id, type_id_t src_type_id,
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
 * to the destination data type, optionally specialized based on
 * the fixed strides provided.
 *
 * This function is only for the built-in data types like int,
 * float, complex. A built-in data type can be detected by
 * checking whether (dtype_instance.extended() == NULL).
 *
 * If a stride is unknown or non-fixed, pass INTPTR_MAX for that stride.
 */
void get_builtin_dtype_assignment_kernel(
                    type_id_t dst_type_id, intptr_t dst_fixedstride,
                    type_id_t src_dt, intptr_t src_fixedstride,
                    assign_error_mode errmode,
                    kernel_instance<unary_operation_t>& out_kernel);

/**
 * Gets a unary kernel for assigning a pod dtype, i.e. a raw
 * byte-copy. It specializes the kernel based on the element size
 * and alignment.
 *
 * If a stride is unknown or non-fixed, pass INTPTR_MAX for that stride.
 */
void get_pod_dtype_assignment_kernel(
                    intptr_t element_size, intptr_t alignment,
                    intptr_t dst_fixedstride, intptr_t src_fixedstride,
                    kernel_instance<unary_operation_t>& out_kernel);

/**
 * Returns a kernel for assigning from the source data type
 * to the destination data type, optionally specialized based on
 * the fixed strides provided.
 *
 * If a stride is unknown or non-fixed, pass INTPTR_MAX for that stride.
 */
void get_dtype_assignment_kernel(
                    const dtype& dst_dt, intptr_t dst_fixedstride,
                    const dtype& src_dt, intptr_t src_fixedstride,
                    assign_error_mode errmode,
                    kernel_instance<unary_operation_t>& out_kernel);

/**
 * Returns a kernel for assigning from the source data to the dest data, with
 * matching source and destination dtypes.
 *
 * If a stride is unknown or non-fixed, pass INTPTR_MAX for that stride.
 */
void get_dtype_assignment_kernel(
                    const dtype& dt,
                    intptr_t dst_fixedstride,
                    intptr_t src_fixedstride,
                    kernel_instance<unary_operation_t>& out_kernel);

} // namespace dnd

#endif // _DND__ASSIGNMENT_KERNELS_HPP_