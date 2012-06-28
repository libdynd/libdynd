//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
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

inline string broadcast_error_message(int dst_ndim, const intptr_t *dst_shape,
                    int src_ndim, const intptr_t *src_shape)
{
    stringstream ss;

    ss << "broadcast error: cannot broadcast shape ";
    print_shape(ss, src_ndim, src_shape);
    ss << " to shape ";
    print_shape(ss, dst_ndim, dst_shape);

    return ss.str();
}

dnd::broadcast_error::broadcast_error(int dst_ndim, const intptr_t *dst_shape,
                    int src_ndim, const intptr_t *src_shape)
    : m_what(broadcast_error_message(dst_ndim, dst_shape, src_ndim, src_shape))
{
}

inline string broadcast_error_message(int noperands, const ndarray **operands)
{
    stringstream ss;

    ss << "broadcast error: cannot broadcast input operand shapes ";
    for (int i = 0; i < noperands; ++i) {
        print_shape(ss, operands[i]->get_ndim(), operands[i]->get_shape());
        if (i != noperands - 1) {
            ss << " ";
        }
    }

    return ss.str();
}

dnd::broadcast_error::broadcast_error(int noperands, const ndarray **operands)
    : m_what(broadcast_error_message(noperands, operands))
{
}

inline string broadcast_error_message(int noperands, ndarray_node **operands)
{
    stringstream ss;

    ss << "broadcast error: cannot broadcast input operand shapes ";
    for (int i = 0; i < noperands; ++i) {
        print_shape(ss, operands[i]->get_ndim(), operands[i]->get_shape());
        if (i != noperands - 1) {
            ss << " ";
        }
    }

    return ss.str();
}

dnd::broadcast_error::broadcast_error(int noperands, ndarray_node **operands)
    : m_what(broadcast_error_message(noperands, operands))
{
}

inline string too_many_indices_message(int nindex, int ndim)
{
    std::stringstream ss;

    ss << "too many indices: provided " << nindex << " indices, but array has only ";
    ss << ndim << " dimensions";

    return ss.str();
}

dnd::too_many_indices::too_many_indices(int nindex, int ndim)
    : m_what(too_many_indices_message(nindex, ndim))
{
    //cout << "throwing too_many_indices\n";
}

inline string index_out_of_bounds_message(intptr_t i, int axis, int ndim, const intptr_t *shape)
{
    stringstream ss;

    ss << "index out of bounds: index " << i << " is out of bounds for axis " << axis;
    ss << " of shape ";
    print_shape(ss, ndim, shape);

    return ss.str();
}

index_out_of_bounds::index_out_of_bounds(intptr_t i, int axis, int ndim, const intptr_t *shape)
    : m_what(index_out_of_bounds_message(i, axis, ndim, shape))
{
}

inline string axis_out_of_bounds_message(intptr_t i, intptr_t start, intptr_t end)
{
    stringstream ss;

    ss << "axis out of bounds: axis " << i << " is not in the half-open range [";
    ss << start << ", " << end << ")";

    return ss.str();
}

dnd::axis_out_of_bounds::axis_out_of_bounds(intptr_t i, intptr_t start, intptr_t end)
    : m_what(axis_out_of_bounds_message(i, start, end))
{
    //cout << "throwing axis_out_of_bounds\n";
}

inline string irange_out_of_bounds_message(const irange& i, int axis, int ndim, const intptr_t *shape)
{
    stringstream ss;

    ss << "irange out of bounds: index range (" << i.start() << " to " << i.finish();
    if (i.step() != 1) {
        ss << " step " << i.step();
    }
    ss << ") is out of bounds for axis " << axis;
    ss << " of shape ";
    print_shape(ss, ndim, shape);

    return ss.str();
}

dnd::irange_out_of_bounds::irange_out_of_bounds(const irange& i, int axis, int ndim, const intptr_t *shape)
    : m_what(irange_out_of_bounds_message(i, axis, ndim, shape))
{
    //cout << "throwing irange_out_of_bounds\n";
}

inline string invalid_type_id_message(int type_id)
{
    stringstream ss;

    ss << "invalid type id: the id " << type_id << " is not valid";

    return ss.str();
}

dnd::invalid_type_id::invalid_type_id(int type_id)
    : m_what(invalid_type_id_message(type_id))
{
    //cout << "throwing invalid_type_id\n";
}

