//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRING_NUMERIC_ASSIGNMENT_KERNELS_HPP_
#define _DYND__STRING_NUMERIC_ASSIGNMENT_KERNELS_HPP_

#include <dynd/dtype.hpp>
#include <dynd/kernels/kernel_instance.hpp>

namespace dynd {

/**
 * Gets a kernel which converts strings to values of a builtin type id.
 */
void get_string_to_builtin_assignment_kernel(type_id_t dst_type_id,
                const dtype& src_string_dt,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel);

void get_builtin_to_string_assignment_kernel(const dtype& dst_string_dt,
                type_id_t src_type_id,
                assign_error_mode errmode,
                kernel_instance<unary_operation_pair_t>& out_kernel);

} // namespace dynd

#endif // _DYND__STRING_NUMERIC_ASSIGNMENT_KERNELS_HPP_
