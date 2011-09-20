//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DTYPE_CASTING_HPP_
#define _DTYPE_CASTING_HPP_

#include <dnd/dtype.hpp>

namespace dnd {

/**
 * An enumeration for controlling what kind of casting is permitted.
 *
 * In an assignment, this pairs up with a set of assign_error_flags,
 * which control whether exceptions are raised on overflow or inexact-ness.
 */
enum dtype_casting {
    /** Permit only exact dtype equality */
    exact_casting,
    /** Permit also byte-order swaps */
    equiv_casting,
    /**
     * Permit casting only when the destination dtype can represent
     * every value of the source dtype.
     */
    lossless_casting,
    /** Permit casting between values of equal kind */
    same_kind_casting,
    /** Permit any kinds of casting whatsoever */
    any_casting
};

/** If the dtypes are exactly the same */
bool can_cast_exact(const dtype& dst_dt, const dtype& src_dt);
/** If the dtypes are exactly the same, up to byte order differences */
bool can_cast_equiv(const dtype& dst_dt, const dtype& src_dt);
/** If 'src' can be cast to 'dst' with no loss of information */
bool can_cast_lossless(const dtype& dst_dt, const dtype& src_dt);
/** If 'src' can be cast to 'dst' without switching to a lesser kind */
bool can_cast_same_kind(const dtype& dst_dt, const dtype& src_dt);

/** Whether the casting is permitted according to the enum rule given */
bool can_cast(const dtype& dst_dt, const dtype& src_dt, dtype_casting rule);

} // namespace dnd

#endif//_DTYPE_CASTING_HPP_
