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

bool can_cast_losslessly(const dtype& dst_dt, const dtype& src_dt);

} // namespace dnd

#endif//_DTYPE_CASTING_HPP_
