//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATE_ASSIGNMENT_KERNELS_HPP_
#define _DYND__DATE_ASSIGNMENT_KERNELS_HPP_

#include <dynd/kernels/kernel_instance.hpp>
#include <dynd/dtypes/date_dtype.hpp>

namespace dynd {

/**
 * Gets a kernel which converts strings to dates.
 */
void get_string_to_date_assignment_kernel(const dtype& src_string_dtype,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel);

/**
 * Gets a kernel which converts dates to strings.
 */
void get_date_to_string_assignment_kernel(const dtype& dst_string_dtype,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel);

/**
 * Gets a kernel which converts dates to structs.
 */
void get_date_to_struct_assignment_kernel(const dtype& dst_struct_dtype,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel);

/**
 * Gets a kernel which converts dates to structs.
 */
void get_struct_to_date_assignment_kernel(const dtype& src_struct_dtype,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel);

/**
 * Makes a kernel which converts strings to dates.
 */
size_t make_string_to_date_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const dtype& src_string_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which converts dates to strings.
 */
size_t make_date_to_string_assignment_kernel(
                hierarchical_kernel<unary_single_operation_t> *out,
                size_t offset_out,
                const dtype& dst_string_dt, const char *dst_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx);

/**
 * This is the layout of the default struct that is converted
 * to/from date dtypes.
 */
struct date_dtype_default_struct {
    int32_t year;
    int8_t month, day;
};

/**
 * This is an array of fixedstruct dtypes, one for each
 * date_unit_t, which is the layout of the default struct
 * converted to/from date dtypes.
 */
extern const dtype date_dtype_default_struct_dtype;

} // namespace dynd

#endif // _DYND__DATE_ASSIGNMENT_KERNELS_HPP_