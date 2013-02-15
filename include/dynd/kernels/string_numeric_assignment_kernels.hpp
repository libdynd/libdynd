//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__STRING_NUMERIC_ASSIGNMENT_KERNELS_HPP_
#define _DYND__STRING_NUMERIC_ASSIGNMENT_KERNELS_HPP_

#include <dynd/dtype.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

namespace dynd {

/**
 * Converts a single UTF8 string to an integer type.
 */
void assign_utf8_string_to_builtin(type_id_t dst_type_id, char *dst,
                const char *str_begin, const char *str_end, assign_error_mode errmode = assign_error_fractional);

/**
 * Makes a kernel which converts strings to values of a builtin type id.
 */
size_t make_builtin_to_string_assignment_kernel(
                assignment_kernel *out,
                size_t offset_out,
                const dtype& dst_string_dt, const char *dst_metadata,
                type_id_t src_type_id,
                assign_error_mode errmode,
                const eval::eval_context *ectx);

/**
 * Makes a kernel which converts values of a builtin type id to strings.
 */
size_t make_string_to_builtin_assignment_kernel(
                assignment_kernel *out,
                size_t offset_out,
                type_id_t dst_type_id,
                const dtype& src_string_dt, const char *src_metadata,
                assign_error_mode errmode,
                const eval::eval_context *ectx);
} // namespace dynd

#endif // _DYND__STRING_NUMERIC_ASSIGNMENT_KERNELS_HPP_
