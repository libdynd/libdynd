//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__DTYPE_ASSIGN_HPP_
#define _DND__DTYPE_ASSIGN_HPP_

#include <utility>

#include <dnd/dtype.hpp>

namespace dnd {

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
    assign_error_inexact
};

const assign_error_mode default_error_mode = assign_error_fractional;

std::ostream& operator<<(std::ostream& o, assign_error_mode errmode);

/** If 'src' can always be cast to 'dst' with no loss of information */
bool is_lossless_assignment(const dtype& dst_dt, const dtype& src_dt);

/** 
 * Assign one element where src and dst may have different dtypes.
 * If the cast can be done losslessly, calls dtype_assign_noexcept,
 * otherwise it will do a checked assignment which may raise
 * an exception.
 *
 * The src and dst data must be aligned. TODO: Relax this restriction.
 */
void dtype_assign(const dtype& dst_dt, char *dst, const dtype& src_dt, const char *src,
                                assign_error_mode errmode = assign_error_fractional);

/**
 * Like dtype_assign, but for strided assignment. Does not require that the data
 * be aligned.
 */
void dtype_strided_assign(const dtype& dst_dt, char *dst, intptr_t dst_stride,
                            const dtype& src_dt, const char *src, intptr_t src_stride,
                            intptr_t count, assign_error_mode errmode);

/**
 * Returns a function for assigning from the source data type
 * to the destination data type, optionally specialized based on
 * the fixed strides provided.
 *
 * If a stride is unknown or non-fixed, pass INTPTR_MAX for that stride.
 */
void get_dtype_strided_assign_operation(
                    const dtype& dst_dt, intptr_t dst_fixedstride,
                    const dtype& src_dt, intptr_t src_fixedstride,
                    assign_error_mode errmode,
                    kernel_instance<unary_operation_t>& out_kernel);

/**
 * Returns a function for assigning from the source data to the dest data, with
 * just one dtype.
 */
void get_dtype_strided_assign_operation(
                    const dtype& dt,
                    intptr_t dst_fixedstride,
                    intptr_t src_fixedstride,
                    kernel_instance<unary_operation_t>& out_kernel);

} // namespace dnd

#endif // _DND__DTYPE_ASSIGN_HPP_
