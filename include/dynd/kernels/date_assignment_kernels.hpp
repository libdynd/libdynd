//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DATE_ASSIGNMENT_KERNELS_HPP_
#define _DYND__DATE_ASSIGNMENT_KERNELS_HPP_

#include <dynd/kernels/unary_kernel_instance.hpp>
#include <dynd/dtypes/date_dtype.hpp>

namespace dynd {

/**
 * Gets a kernel which converts strings to dates.
 */
void get_string_to_date_assignment_kernel(date_unit_t dst_unit,
                const dtype& src_string_dtype,
                unary_specialization_kernel_instance& out_kernel);

} // namespace dynd

#endif // _DYND__DATE_ASSIGNMENT_KERNELS_HPP_