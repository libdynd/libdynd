//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__SHAPE_TOOLS_HPP_
#define _DND__SHAPE_TOOLS_HPP_

#include <stdint.h>

namespace dnd {

/**
 * This function broadcasts the dimensions and strides of 'src' to a given
 * shape, raising an error if it cannot be broadcast.
 */
void broadcast_to_shape(int ndim, const intptr_t *shape,
                int src_ndim, const intptr_t *src_shape, const intptr_t *src_strides,
                intptr_t *out_strides);

} // namespace dnd

#endif // _DND__SHAPE_TOOLS_HPP_
