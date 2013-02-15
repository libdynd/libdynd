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
 * Creates an assignment kernel for one data value from the
 * src dtype/metadata to the dst dtype/metadata. This adds the
 * kernel at the 'out_offset' position in 'out's data, as part
 * of a hierarchy matching the dtype's hierarchy.
 *
 * This function should always be called with this == dst_dt first,
 * and dtypes which don't support the particular assignment should
 * then call the corresponding function with this == src_dt.
 */
size_t make_assignment_kernel(
                assignment_kernel *out,
                size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx);

/**
 * Creates an assignment kernel when the src and the dst are the same,
 * and are POD (plain old data).
 */
size_t make_pod_dtype_assignment_kernel(
                assignment_kernel *out,
                size_t offset_out,
                size_t data_size, size_t data_alignment);

/**
 * Creates an assignment kernel from the src to the dst built in
 * type ids.
 */
size_t make_builtin_dtype_assignment_function(
                assignment_kernel *out,
                size_t offset_out,
                type_id_t dst_type_id, type_id_t src_type_id,
                assign_error_mode errmode);

/**
 * Generic assignment kernel + destructor for a strided dimension.
 */
struct strided_assign_kernel_extra {
    typedef strided_assign_kernel_extra extra_type;

    kernel_data_prefix base;
    intptr_t size;
    intptr_t dst_stride, src_stride;

    static void single(char *dst, const char *src,
                        kernel_data_prefix *extra);
    static void destruct(kernel_data_prefix *extra);
};

} // namespace dynd

#endif // _DYND__ASSIGNMENT_KERNELS_HPP_
