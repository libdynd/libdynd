//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <dnd/exceptions.hpp>

#include <sstream>

using namespace std;
using namespace dnd;

dnd::broadcast_error::broadcast_error(int dst_ndim, const intptr_t *dst_shape,
                    int src_ndim, const intptr_t *src_shape) {
    std::stringstream ss;
    ss << "broadcast error: cannot broadcast shape (";
    for (int i = 0; i < src_ndim; ++i) {
        ss << src_shape[i];
        if (i != src_ndim - 1) {
            ss << " ";
        }
    }
    ss << ") to shape (";
    for (int i = 0; i < dst_ndim; ++i) {
        ss << dst_shape[i];
        if (i != dst_ndim - 1) {
            ss << " ";
        }
    }
    ss << ")";

    m_what = ss.str();
}

