//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <dnd/shape_tools.hpp>
#include <dnd/exceptions.hpp>

void dnd::broadcast_to_shape(int dst_ndim, const intptr_t *dst_shape,
                int src_ndim, const intptr_t *src_shape, const intptr_t *src_strides,
                intptr_t *out_strides)
{
    if (src_ndim > dst_ndim) {
        throw broadcast_error(dst_ndim, dst_shape, src_ndim, src_shape);
    }

    int dimdelta = dst_ndim - src_ndim;
    for (int i = 0; i < dimdelta; ++i) {
        out_strides[i] = 0;
    }
    for (int i = dimdelta; i < dst_ndim; ++i) {
        int src_i = i - dimdelta;
        if (src_shape[src_i] == 1) {
            out_strides[i] = 0;
        } else if (src_shape[src_i] != dst_shape[i]) {
            throw broadcast_error(dst_ndim, dst_shape, src_ndim, src_shape);
        } else {
            out_strides[i] = src_strides[src_i];
        }
    }
}
