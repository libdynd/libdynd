//
// Copyright (C) 2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ARRAY_ASSIGNMENT_KERNELS_HPP_
#define _DYND__ARRAY_ASSIGNMENT_KERNELS_HPP_

#include <dynd/dtype.hpp>
#include <dynd/kernels/hierarchical_kernels.hpp>

namespace dynd {

/**
 * Makes a kernel which broadcasts the input to a var array
 */
size_t make_broadcast_to_var_array_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const dtype& dst_var_array_dt, const char *dst_metadata,
                const dtype& src_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which assigns var arrays
 */
size_t make_var_array_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const dtype& dst_var_array_dt, const char *dst_metadata,
                const dtype& src_var_array_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which assigns strided arrays to var arrays
 */
size_t make_strided_to_var_array_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const dtype& dst_var_array_dt, const char *dst_metadata,
                const dtype& src_strided_array_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which assigns var arrays to strided arrays
 */
size_t make_var_to_strided_array_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const dtype& dst_strided_array_dt, const char *dst_metadata,
                const dtype& src_var_array_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__ARRAY_ASSIGNMENT_KERNELS_HPP_
