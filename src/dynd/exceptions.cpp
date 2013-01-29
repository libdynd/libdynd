//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>
#include <iostream> // for DEBUG

#include <dynd/exceptions.hpp>
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

broadcast_error::broadcast_error(int dst_ndim, const intptr_t *dst_shape,
                    int src_ndim, const intptr_t *src_shape)
    : dynd_exception("broadcast error", broadcast_error_message(dst_ndim, dst_shape, src_ndim, src_shape))
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

broadcast_error::broadcast_error(const ndobject& dst, const ndobject& src)
    : dynd_exception("broadcast error", broadcast_error_message(dst, src))
{
}

inline string broadcast_error_message(size_t ninputs, const ndobject* inputs)
{
    stringstream ss;

    ss << "cannot broadcast input dynd operands with shapes ";
    for (size_t i = 0; i < ninputs; ++i) {
        size_t undim = inputs[i].get_undim();
        dimvector shape(undim);
        inputs[i].get_shape(shape.get());
        print_shape(ss, undim, shape.get());
        if (i + 1 != ninputs) {
            ss << " ";
        }
    }

    return ss.str();
}

broadcast_error::broadcast_error(size_t ninputs, const ndobject *inputs)
    : dynd_exception("broadcast error", broadcast_error_message(ninputs, inputs))
{
}


inline string too_many_indices_message(const dtype& dt, size_t nindices, size_t ndim)
{
    std::stringstream ss;

    ss << "provided " << nindices << " indices to dynd dtype " << dt << ", but only ";
    ss << ndim << " dimensions available";

    return ss.str();
}

dynd::too_many_indices::too_many_indices(const dtype& dt, size_t nindices, size_t ndim)
    : dynd_exception("too many indices", too_many_indices_message(dt, nindices, ndim))
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

inline string index_out_of_bounds_message(intptr_t i, intptr_t dimension_size)
{
    stringstream ss;

    ss << "index " << i << " is out of bounds for dimension of size " << dimension_size;

    return ss.str();
}

index_out_of_bounds::index_out_of_bounds(intptr_t i, size_t axis, size_t ndim, const intptr_t *shape)
    : dynd_exception("index out of bounds", index_out_of_bounds_message(i, axis, ndim, shape))
{
}

index_out_of_bounds::index_out_of_bounds(intptr_t i, size_t axis, const std::vector<intptr_t>& shape)
    : dynd_exception("index out of bounds", index_out_of_bounds_message(i, axis, (int)shape.size(), shape.empty() ? NULL : &shape[0]))
{
}

index_out_of_bounds::index_out_of_bounds(intptr_t i, intptr_t dimension_size)
    : dynd_exception("index out of bounds", index_out_of_bounds_message(i, dimension_size))
{
}

inline string axis_out_of_bounds_message(size_t i, size_t ndim)
{
    stringstream ss;

    ss << "axis " << i << " is not a valid axis for an " << ndim << " dimensional operation";

    return ss.str();
}

dynd::axis_out_of_bounds::axis_out_of_bounds(size_t i, size_t ndim)
    : dynd_exception("axis out of bounds", axis_out_of_bounds_message(i, ndim))
{
    //cout << "throwing axis_out_of_bounds\n";
}

inline void print_slice(std::ostream& o, const irange& i)
{
    if (i.step() == 0) {
        o << '[' << i.start() << ']';
    } else {
        o << '[';
        if (i.start() != std::numeric_limits<intptr_t>::min()) {
            o << i.start();
        }
        o << ':';
        if (i.finish() != std::numeric_limits<intptr_t>::max()) {
            o << i.finish();
        }
        if (i.step() != 1) {
            o << ':';
            o << i.step();
        }
        o << ']';
    }
}

inline string irange_out_of_bounds_message(const irange& i, size_t axis, size_t ndim, const intptr_t *shape)
{
    stringstream ss;

    ss << "index range ";
    print_slice(ss, i);
    ss << " is out of bounds for axis " << axis;
    ss << " in shape ";
    print_shape(ss, ndim, shape);

    return ss.str();
}

inline string irange_out_of_bounds_message(const irange& i, intptr_t dimension_size)
{
    stringstream ss;

    ss << "index range ";
    print_slice(ss, i);
    ss << " is out of bounds for dimension of size " << dimension_size;

    return ss.str();
}

irange_out_of_bounds::irange_out_of_bounds(const irange& i, size_t axis, size_t ndim, const intptr_t *shape)
    : dynd_exception("irange out of bounds", irange_out_of_bounds_message(i, axis, ndim, shape))
{
    //cout << "throwing irange_out_of_bounds\n";
}

irange_out_of_bounds::irange_out_of_bounds(const irange& i, size_t axis, const std::vector<intptr_t>& shape)
    : dynd_exception("irange out of bounds", irange_out_of_bounds_message(i, axis, (int)shape.size(), shape.empty() ? NULL : &shape[0]))
{
}

irange_out_of_bounds::irange_out_of_bounds(const irange& i, intptr_t dimension_size)
    : dynd_exception("irange out of bounds", irange_out_of_bounds_message(i, dimension_size))
{
}


inline string invalid_type_id_message(int type_id)
{
    stringstream ss;

    ss << "the id " << type_id << " is not valid";

    return ss.str();
}

dynd::invalid_type_id::invalid_type_id(int type_id)
    : dynd_exception("invalid type id", invalid_type_id_message(type_id))
{
    //cout << "throwing invalid_type_id\n";
}

