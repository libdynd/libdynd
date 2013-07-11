//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DTYPE_ASSIGN_HPP_
#define _DYND__DTYPE_ASSIGN_HPP_

#include <utility>
#include <iostream>

namespace dynd {

namespace ndt {
    class type;
} // namespace ndt

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
bool is_lossless_assignment(const ndt::type& dst_dt, const ndt::type& src_dt);

/**
 * Copies a value from one location to another, where the dtypes of the source
 * and destination are the same.
 *
 * \param dt  The dtype for the copy operation.
 * \param dst_metadata  The metadata of the destination.
 * \param dst_data  The data where the destination element is stored.
 * \param src_metadata  The metadata of the source.
 * \param src_data  The data where the source element is stored.
 */
void dtype_copy(const ndt::type& dt, const char *dst_metadata, char *dst_data,
                const char *src_metadata, const char *src_data);

/** 
 * Assign one element where src and dst may have different dtypes.
 * Requires that the data be aligned. To assign unaligned data,
 * use make_unaligned_type().
 */
void dtype_assign(const ndt::type& dst_dt, const char *dst_metadata, char *dst_data,
                const ndt::type& src_dt, const char *src_metadata, const char *src_data,
                assign_error_mode errmode = assign_error_fractional,
                const eval::eval_context *ectx = &eval::default_eval_context);

} // namespace dynd

#endif // _DYND__DTYPE_ASSIGN_HPP_
