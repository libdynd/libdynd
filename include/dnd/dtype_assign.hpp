//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DTYPE_ASSIGN_HPP_
#define _DTYPE_ASSIGN_HPP_

#include <utility>

#include <dnd/dtype.hpp>
#include <dnd/dtype_casting.hpp>

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

/** A base class for auxiliary data used by the unary_operation function pointers. */
class auxiliary_data {
public:

    virtual ~auxiliary_data() {
    }
};

/**
 * Assign one element where src and dst may have different dtypes.
 * If the cast can be done losslessly, calls dtype_assign_noexcept,
 * otherwise it will do a checked assignment which may raise
 * an exception.
 *
 * The src and dst data must be aligned. TODO: Relax this restriction.
 */
void dtype_assign(const dtype& dst_dt, void *dst, const dtype& src_dt, const void *src,
                                assign_error_mode errmode = assign_error_fractional);

/**
 * Like dtype_assign, but for strided assignment. Does not require that the data
 * be aligned.
 */
void dtype_strided_assign(const dtype& dst_dt, void *dst, intptr_t dst_stride,
                            const dtype& src_dt, const void *src, intptr_t src_stride,
                            intptr_t count, assign_error_mode errmode);

/**
 * The function pointer type for a unary operation, for example a casting function
 * from one dtype to another.
 */
typedef void (*unary_operation_t)(void *dst, intptr_t dst_stride,
                                const void *src, intptr_t src_stride,
                                intptr_t count,
                                const auxiliary_data *auxdata);

/**
 * Returns a function for assigning from the source data type
 * to the destination data type, optionally specialized based on
 * the fixed strides provided.
 *
 * If a stride is unknown or non-fixed, pass INTPTR_MAX for that stride.
 *
 * Pass the bitwise-OR (|) of all the input array strides and origin pointers
 * of both src and dst to align_test. If this is not possible,
 * pass the value 1 to indicate the data may be aligned or not,
 * or the value 0 to indicate the data is definitely aligned.
 */
std::pair<unary_operation_t, std::shared_ptr<auxiliary_data> > get_dtype_strided_assign_operation(
                    const dtype& dst_dt, intptr_t dst_fixedstride, char dst_align_test,
                    const dtype& src_dt, intptr_t src_fixedstride, char src_align_test,
                    assign_error_mode errmode);

/**
 * Returns a function for assigning from the source data to the dest data, with
 * a fixed data type.
 */
std::pair<unary_operation_t, std::shared_ptr<auxiliary_data> > get_dtype_strided_assign_operation(
                    const dtype& dt,
                    intptr_t dst_fixedstride, char dst_align_test,
                    intptr_t src_fixedstride, char src_align_test);

} // namespace dnd

#endif//_DTYPE_ASSIGN_HPP_
