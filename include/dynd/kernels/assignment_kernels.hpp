//
// Copyright (C) 2011-13, DyND Developers
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

/**
 * Creates an assignment kernel for one data value from the
 * src dtype/metadata to the dst dtype/metadata. This adds the
 * kernel at the 'out_offset' position in 'out's data, as part
 * of a hierarchy matching the dtype's hierarchy.
 *
 * This function should always be called with this == dst_dt first,
 * and dtypes which don't support the particular assignment should
 * then call the corresponding function with this == src_dt.
 */
void make_assignment_kernel(
                    hierarchical_kernel<unary_single_operation_t> *out,
                    size_t out_offset,
                    const dtype& dst_dt, const char *dst_metadata,
                    const dtype& src_dt, const char *src_metadata,
                    assign_error_mode errmode,
                    const eval::eval_context *ectx);

/**
 * Creates an assignment kernel when the src and the dst are the same,
 * and are POD (plain old data).
 */
void make_pod_dtype_assignment_kernel(
                    hierarchical_kernel<unary_single_operation_t> *out,
                    size_t out_offset,
                    size_t data_size, size_t data_alignment);

/**
 * Creates an assignment kernel from the src to the dst built in
 * type ids.
 */
void make_builtin_dtype_assignment_function(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t out_offset,
                type_id_t dst_type_id, type_id_t src_type_id,
                assign_error_mode errmode);

/**
 * Generic assignment kernel + destructor for a strided dimension.
 */
struct strided_assign_kernel_extra {
    hierarchical_kernel_common_base base;
    intptr_t size;
    intptr_t dst_stride, src_stride;

    static void single(char *dst, const char *src,
                        hierarchical_kernel_common_base *extra);
    static void destruct(hierarchical_kernel_common_base *extra);
};

} // namespace dynd

#endif // _DYND__ASSIGNMENT_KERNELS_HPP_
