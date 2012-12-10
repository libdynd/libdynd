//
// Copyright (C) 2011-12, Dynamic NDArray Developers
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
void get_string_to_date_assignment_kernel(date_unit_t dst_unit,
                const dtype& src_string_dtype,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel);

/**
 * Gets a kernel which converts dates to strings.
 */
void get_date_to_string_assignment_kernel(const dtype& dst_string_dtype,
                date_unit_t src_unit,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel);

/**
 * Gets a kernel which converts dates to structs.
 */
void get_date_to_struct_assignment_kernel(const dtype& dst_struct_dtype,
                date_unit_t src_unit,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel);

} // namespace dynd

#endif // _DYND__DATE_ASSIGNMENT_KERNELS_HPP_