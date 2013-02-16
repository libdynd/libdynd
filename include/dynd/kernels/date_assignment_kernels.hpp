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
 * Makes a kernel which converts strings to dates.
 */
size_t make_string_to_date_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                const dtype& src_string_dt, const char *src_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which converts dates to strings.
 */
size_t make_date_to_string_assignment_kernel(
                assignment_kernel *out, size_t offset_out,
                const dtype& dst_string_dt, const char *dst_metadata,
                kernel_request_t kernreq, assign_error_mode errmode,
                const eval::eval_context *ectx);

} // namespace dynd

#endif // _DYND__DATE_ASSIGNMENT_KERNELS_HPP_