//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <sstream>
#include <iostream> // for DEBUG

#include <dnd/exceptions.hpp>

using namespace std;
using namespace dnd;

dnd::broadcast_error::broadcast_error(int dst_ndim, const intptr_t *dst_shape,
                    int src_ndim, const intptr_t *src_shape)
{
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

dnd::too_many_indices::too_many_indices(int nindex, int ndim)
{
    //cout << "throwing too_many_indices\n";
    std::stringstream ss;

    ss << "too many indices: provided " << nindex << " indices, but object has only ";
    ss << ndim << " dimensions";

    m_what = ss.str();
}

dnd::index_out_of_bounds::index_out_of_bounds(intptr_t i, intptr_t start, intptr_t end)
{
    //cout << "throwing index_out_of_bounds\n";
    std::stringstream ss;

    ss << "index out of bounds: index " << i << " is not in the half-open range [";
    ss << start << ", " << end << ")";

    m_what = ss.str();
}

dnd::irange_out_of_bounds::irange_out_of_bounds(const irange& i, intptr_t start, intptr_t end)
{
    //cout << "throwing irange_out_of_bounds\n";
    std::stringstream ss;

    ss << "irange out of bounds: index range (" << i.start() << " to " << i.finish();
    if (i.step() != 1) {
        ss << " step " << i.step();
    }
    ss << ") is not in the half-open range [";
    ss << start << ", " << end << ")";

    m_what = ss.str();
}

