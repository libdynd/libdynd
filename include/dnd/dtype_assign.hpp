//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DTYPE_ASSIGN_HPP_
#define _DTYPE_ASSIGN_HPP_

#include <dnd/dtype.hpp>

namespace dnd {

// Assign one element where src and dst may have different dtypes.
// If the cast can be done losslessly, calls dtype_assign_noexcept,
// otherwise it will do a checked assignment which may raise
// an exception.
void dtype_assign(void *dst, const void *src, dtype dst_dt, dtype src_dt);

// Assign one element where src and dst may have different dtypes.
// This function does lossy casts if necessary without raising an
// exception.
void dtype_assign_noexcept(void *dst, const void *src, dtype dst_dt, dtype src_dt);

// Assign 

} // namespace dnd

#endif//_DTYPE_ASSIGN_HPP_
