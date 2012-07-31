//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__DTYPE_ASSIGN_HPP_
#define _DND__DTYPE_ASSIGN_HPP_

#include <utility>

namespace dnd {

class dtype;
namespace eval {
    struct eval_context;
    extern const eval_context default_eval_context;
} // namespace eval

/**
 * An enumeration for the error checks during assignment.
 */
enum assign_error_mode {
    /** No error checking during assignment */
    assign_error_none,
    /** Overflow checking, but loss of precision ok. This checks loss of imaginary component  */
    assign_error_overflow,
    /** Overflow and loss of fractional part (for float -> int) checking */
    assign_error_fractional,
    /** Overflow and floating point precision loss checking */
    assign_error_inexact,
    /** Use the mode specified in the eval_context */
    assign_error_default
};

std::ostream& operator<<(std::ostream& o, assign_error_mode errmode);

/** If 'src' can always be cast to 'dst' with no loss of information */
bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt);

/** 
 * Assign one element where src and dst may have different dtypes.
 * Requires that the data be aligned. To assign unaligned data,
 * use make_unaligned_dtype().
 */
void dtype_assign(const dtype& dst_dt, char *dst, const dtype& src_dt, const char *src,
                                assign_error_mode errmode = assign_error_fractional,
                                const eval::eval_context *ectx = &eval::default_eval_context);

/**
 * Like dtype_assign, but for strided assignment. Requires that the data
 * be aligned. To assign unaligned data, use make_unaligned_dtype().
 */
void dtype_strided_assign(const dtype& dst_dt, char *dst, intptr_t dst_stride,
                            const dtype& src_dt, const char *src, intptr_t src_stride,
                            intptr_t count, assign_error_mode errmode,
                            const eval::eval_context *ectx = &eval::default_eval_context);


} // namespace dnd

#endif // _DND__DTYPE_ASSIGN_HPP_
