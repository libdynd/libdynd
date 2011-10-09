//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <sstream>
#include <iostream> // for DEBUG

#include <dnd/exceptions.hpp>
#include <dnd/ndarray.hpp>

using namespace std;
using namespace dnd;

static void print_shape(std::ostream& o, int ndim, const intptr_t *shape)
{
    o << "(";
    for (int i = 0; i < ndim; ++i) {
        o << shape[i];
        if (i != ndim - 1) {
            o << ", ";
        }
    }
    o << ")";
}

dnd::broadcast_error::broadcast_error(int dst_ndim, const intptr_t *dst_shape,
                    int src_ndim, const intptr_t *src_shape)
{
    stringstream ss;

    ss << "broadcast error: cannot broadcast shape ";
    print_shape(ss, src_ndim, src_shape);
    ss << " to shape ";
    print_shape(ss, dst_ndim, dst_shape);

    m_what = ss.str();
}

dnd::broadcast_error::broadcast_error(int noperands, const ndarray **operands)
{
    stringstream ss;

    ss << "broadcast error: cannot broadcast input operand shapes ";
    for (int i = 0; i < noperands; ++i) {
        print_shape(ss, operands[i]->get_ndim(), operands[i]->get_shape());
        if (i != noperands - 1) {
            ss << " ";
        }
    }

    m_what = ss.str();
}

dnd::broadcast_error::broadcast_error(int noperands, ndarray_expr_node **operands)
{
    stringstream ss;

    ss << "broadcast error: cannot broadcast input operand shapes ";
    for (int i = 0; i < noperands; ++i) {
        print_shape(ss, operands[i]->get_ndim(), operands[i]->get_shape());
        if (i != noperands - 1) {
            ss << " ";
        }
    }

    m_what = ss.str();
}

dnd::too_many_indices::too_many_indices(int nindex, int ndim)
{
    //cout << "throwing too_many_indices\n";
    std::stringstream ss;

    ss << "too many indices: provided " << nindex << " indices, but array has only ";
    ss << ndim << " dimensions";

    m_what = ss.str();
}

dnd::index_out_of_bounds::index_out_of_bounds(intptr_t i, intptr_t start, intptr_t end)
{
    //cout << "throwing index_out_of_bounds\n";
    stringstream ss;

    ss << "index out of bounds: index " << i << " is not in the half-open range [";
    ss << start << ", " << end << ")";

    m_what = ss.str();
}

dnd::axis_out_of_bounds::axis_out_of_bounds(intptr_t i, intptr_t start, intptr_t end)
{
    //cout << "throwing axis_out_of_bounds\n";
    stringstream ss;

    ss << "axis out of bounds: axis " << i << " is not in the half-open range [";
    ss << start << ", " << end << ")";

    m_what = ss.str();
}

dnd::irange_out_of_bounds::irange_out_of_bounds(const irange& i, intptr_t start, intptr_t end)
{
    //cout << "throwing irange_out_of_bounds\n";
    stringstream ss;

    ss << "irange out of bounds: index range (" << i.start() << " to " << i.finish();
    if (i.step() != 1) {
        ss << " step " << i.step();
    }
    ss << ") is not in the half-open range [";
    ss << start << ", " << end << ")";

    m_what = ss.str();
}

