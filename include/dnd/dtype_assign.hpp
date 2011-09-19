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

namespace dnd {

/**
 * A set of flags for what kind of errors to raise when assigning
 * values from one dtype to another.
 */
enum assign_error_flags {
    assign_error_overflow = 1,
    assign_error_inexact = 2
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
 * The src and dst data must be aligned.
 */
void dtype_assign(const dtype& dst_dt, void *dst, const dtype& src_dt, const void *src);

// Assign one element where src and dst may have different dtypes.
// This function does lossy casts if necessary without raising an
// exception.
//
// The src and dst data must be aligned.
void dtype_assign_noexcept(const dtype& dst_dt, void *dst, const dtype& src_dt, const void *src);

// Like dtype_assign, but for strided assignment
void dtype_strided_assign(const dtype& dst_dt, void *dst, intptr_t dst_stride,
                            const dtype& src_dt, const void *src, intptr_t src_stride,
                            intptr_t count);

// Like dtype_assign_noexcept, but for strided assignment
void dtype_strided_assign_noexcept(const dtype& dst_dt, void *dst, intptr_t dst_stride,
                            const dtype& src_dt, const void *src, intptr_t src_stride,
                            intptr_t count);

// The function pointer type for a unary operation, for example a casting function
// from one dtype to another.
typedef void (*unary_operation_t)(void *dst, intptr_t dst_stride,
                                const void *src, intptr_t src_stride,
                                intptr_t count,
                                auxiliary_data *auxdata);

// Returns a function for assigning from the source data type
// to the destination data type, optionally specialized based on
// the fixed strides provided.
//
// If a stride is unknown or non-fixed, pass INTPTR_MAX for that stride.
//
// Pass the bitwise-OR (|) of all the input array strides and origin pointers
// of both src and dst to align_test. If this is not possible,
// pass the value 1 to indicate the data may be aligned or not,
// or the value 0 to indicate the data is definitely aligned.
std::pair<unary_operation_t, auxiliary_data *> get_dtype_strided_assign_operation(
                    const dtype& dst_dt, intptr_t dst_fixedstride,
                    const dtype& src_dt, intptr_t src_fixedstride,
                    char align_test);

std::pair<unary_operation_t, auxiliary_data *> get_dtype_strided_assign_noexcept_operation(
                    const dtype& dst_dt, intptr_t dst_fixedstride,
                    const dtype& src_dt, intptr_t src_fixedstride,
                    char align_test);

} // namespace dnd

#endif//_DTYPE_ASSIGN_HPP_
