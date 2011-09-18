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

bool can_cast_losslessly(const dtype& dst_dt, const dtype& src_dt);

} // namespace dnd

#endif//_DTYPE_CASTING_HPP_
