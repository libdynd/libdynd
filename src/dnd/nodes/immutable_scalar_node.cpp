//
// Copyright (C) 2011-2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/nodes/immutable_scalar_node.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>

using namespace std;
using namespace dnd;

dnd::immutable_scalar_node::immutable_scalar_node(const dtype& dt, const char* data)
    : ndarray_expr_node(dt, 0, 0, NULL, strided_array_node_category, immutable_scalar_node_type)
{
    if (dt.element_size() > (intptr_t)sizeof(m_data)) {
        stringstream ss;
        ss << "input scalar dtype " << dt << " is too large for the immutable scalar node's internal buffer";
        throw runtime_error(ss.str());
    }

    memcpy(&m_data[0], data, dt.element_size());
}

immutable_scalar_node::immutable_scalar_node(const dtype& dt, const char* data, int ndim, const intptr_t *shape)
    : ndarray_expr_node(dt, ndim, 0, shape, strided_array_node_category, immutable_scalar_node_type)
{
    if (dt.element_size() > (intptr_t)sizeof(m_data)) {
        stringstream ss;
        ss << "input scalar dtype " << dt << " is too large for the immutable scalar node's internal buffer";
        throw runtime_error(ss.str());
    }

    memcpy(&m_data[0], data, dt.element_size());
}



void dnd::immutable_scalar_node::as_readwrite_data_and_strides(char **DND_UNUSED(out_originptr),
                                                    intptr_t *DND_UNUSED(out_strides)) const
{
    throw std::runtime_error("cannot write to an immutable scalar dnd::ndarray node");
}

void dnd::immutable_scalar_node::as_readonly_data_and_strides(char const **out_originptr,
                                                    intptr_t *out_strides) const
{
    *out_originptr = reinterpret_cast<const char *>(&m_data[0]);
    for (int i = 0; i < m_ndim; ++i) {
        out_strides[i] = 0;
    }
}

ndarray_expr_node_ptr dnd::immutable_scalar_node::as_dtype(const dtype& dt,
                    dnd::assign_error_mode errmode, bool allow_in_place)
{
    if (allow_in_place) {
        m_dtype = make_conversion_dtype(dt, m_dtype, errmode);
        return ndarray_expr_node_ptr(this);
    } else {
        return ndarray_expr_node_ptr(new immutable_scalar_node(
                        make_conversion_dtype(dt, m_dtype, errmode),
                        reinterpret_cast<const char *>(&m_data[0])));
    }
}

ndarray_expr_node_ptr dnd::immutable_scalar_node::broadcast_to_shape(int ndim,
                    const intptr_t *shape, bool allow_in_place)
{
    // If the shape is identical, don't make a new node
    if (ndim == m_ndim && memcmp(m_shape.get(), shape, ndim * sizeof(intptr_t)) == 0) {
        return ndarray_expr_node_ptr(this);
    }

    if (allow_in_place) {
        if (ndim == m_ndim) {
            memcpy(m_shape.get(), shape, ndim * sizeof(intptr_t));
        } else {
            // Overwrite the shape
            m_shape.init(ndim, shape);
            m_ndim = ndim;
        }
        return ndarray_expr_node_ptr(this);
    } else {
        ndarray_expr_node_ptr result(
                new immutable_scalar_node(m_dtype, reinterpret_cast<const char *>(m_data), ndim, shape));
        return result;
    }
}


ndarray_expr_node_ptr dnd::immutable_scalar_node::apply_linear_index(
                int ndim, const intptr_t *DND_UNUSED(shape),
                const int *DND_UNUSED(axis_map), const intptr_t *DND_UNUSED(index_strides),
                const intptr_t *DND_UNUSED(start_index), bool DND_UNUSED(allow_in_place))
{
    if (ndim == 0) {
        return ndarray_expr_node_ptr(this);
    } else {
        throw runtime_error("cannot index into an immutable scalar dnd::ndarray node");
    }
}

void dnd::immutable_scalar_node::debug_dump_extra(std::ostream& o, const std::string& indent) const
{
    o << indent << " data: ";
    hexadecimal_print(o, reinterpret_cast<const char *>(&m_data[0]), m_dtype.element_size());
    o << "\n";
}
