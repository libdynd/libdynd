//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>
#include <iostream> // for DEBUG

#include <dynd/exceptions.hpp>
#include <dynd/ndarray.hpp>
#include <dynd/ndobject.hpp>
#include <dynd/shape_tools.hpp>

using namespace std;
using namespace dynd;

inline string broadcast_error_message(int dst_ndim, const intptr_t *dst_shape,
                    int src_ndim, const intptr_t *src_shape)
{
    stringstream ss;

    ss << "cannot broadcast shape ";
    print_shape(ss, src_ndim, src_shape);
    ss << " to shape ";
    print_shape(ss, dst_ndim, dst_shape);

    return ss.str();
}

dynd::broadcast_error::broadcast_error(int dst_ndim, const intptr_t *dst_shape,
                    int src_ndim, const intptr_t *src_shape)
    : dnd_exception("broadcast error", broadcast_error_message(dst_ndim, dst_shape, src_ndim, src_shape))
{
}

inline string broadcast_error_message(const ndobject& dst, const ndobject& src)
{
    vector<intptr_t> dst_shape = dst.get_shape(), src_shape = src.get_shape();
    stringstream ss;

    ss << "cannot broadcast ndobject with dtype ";
    ss << src.get_dtype() << " and shape ";
    print_shape(ss, src_shape);
    ss << " to dtype " << dst.get_dtype() << " and shape ";
    print_shape(ss, dst_shape);

    return ss.str();
}

dynd::broadcast_error::broadcast_error(const ndobject& dst, const ndobject& src)
    : dnd_exception("broadcast error", broadcast_error_message(dst, src))
{
}


inline string broadcast_error_message(int noperands, const ndarray_node_ptr *operands)
{
    stringstream ss;

    ss << "cannot broadcast input operand shapes ";
    for (int i = 0; i < noperands; ++i) {
        print_shape(ss, operands[i]->get_ndim(), operands[i]->get_shape());
        if (i != noperands - 1) {
            ss << " ";
        }
    }

    return ss.str();
}

dynd::broadcast_error::broadcast_error(int noperands, const ndarray_node_ptr *operands)
    : dnd_exception("broadcast error", broadcast_error_message(noperands, operands))
{
}

inline string too_many_indices_message(int nindex, int ndim)
{
    std::stringstream ss;

    ss << "provided " << nindex << " indices, but array has only ";
    ss << ndim << " dimensions";

    return ss.str();
}

dynd::too_many_indices::too_many_indices(int nindex, int ndim)
    : dnd_exception("too many indices", too_many_indices_message(nindex, ndim))
{
    //cout << "throwing too_many_indices\n";
}

inline string index_out_of_bounds_message(intptr_t i, int axis, int ndim, const intptr_t *shape)
{
    stringstream ss;

    ss << "index " << i << " is out of bounds for axis " << axis;
    ss << " in shape ";
    print_shape(ss, ndim, shape);

    return ss.str();
}

index_out_of_bounds::index_out_of_bounds(intptr_t i, int axis, int ndim, const intptr_t *shape)
    : dnd_exception("index out of bounds", index_out_of_bounds_message(i, axis, ndim, shape))
{
}

index_out_of_bounds::index_out_of_bounds(intptr_t i, int axis, const std::vector<intptr_t>& shape)
    : dnd_exception("index out of bounds", index_out_of_bounds_message(i, axis, (int)shape.size(), shape.empty() ? NULL : &shape[0]))
{
}


inline string axis_out_of_bounds_message(intptr_t i, intptr_t ndim)
{
    stringstream ss;

    ss << "axis " << i << " is not a valid axis for an " << ndim << " dimensional operation";

    return ss.str();
}

dynd::axis_out_of_bounds::axis_out_of_bounds(intptr_t i, intptr_t ndim)
    : dnd_exception("axis out of bounds", axis_out_of_bounds_message(i, ndim))
{
    //cout << "throwing axis_out_of_bounds\n";
}

inline string irange_out_of_bounds_message(const irange& i, int axis, int ndim, const intptr_t *shape)
{
    stringstream ss;

    ss << "index range (" << i.start() << " to " << i.finish();
    if (i.step() != 1) {
        ss << " step " << i.step();
    }
    ss << ") is out of bounds for axis " << axis;
    ss << " in shape ";
    print_shape(ss, ndim, shape);

    return ss.str();
}

dynd::irange_out_of_bounds::irange_out_of_bounds(const irange& i, int axis, int ndim, const intptr_t *shape)
    : dnd_exception("irange out of bounds", irange_out_of_bounds_message(i, axis, ndim, shape))
{
    //cout << "throwing irange_out_of_bounds\n";
}

dynd::irange_out_of_bounds::irange_out_of_bounds(const irange& i, int axis, const std::vector<intptr_t>& shape)
    : dnd_exception("irange out of bounds", irange_out_of_bounds_message(i, axis, (int)shape.size(), shape.empty() ? NULL : &shape[0]))
{
}


inline string invalid_type_id_message(int type_id)
{
    stringstream ss;

    ss << "the id " << type_id << " is not valid";

    return ss.str();
}

dynd::invalid_type_id::invalid_type_id(int type_id)
    : dnd_exception("invalid type id", invalid_type_id_message(type_id))
{
    //cout << "throwing invalid_type_id\n";
}

