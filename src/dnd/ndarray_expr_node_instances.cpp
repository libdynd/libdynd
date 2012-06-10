//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <stdexcept>
#include <sstream>

#include <dnd/ndarray.hpp>
#include <dnd/shape_tools.hpp>
#include <dnd/exceptions.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>

#include "ndarray_expr_node_instances.hpp"

using namespace std;
using namespace dnd;


// broadcast_shape_expr_node

dnd::broadcast_shape_expr_node::broadcast_shape_expr_node(int ndim, const intptr_t *shape,
                                                const ndarray_expr_node_ptr& op)
    : ndarray_expr_node(op->get_dtype(), ndim, 1, shape,
        op->get_node_category() == strided_array_node_category ? strided_array_node_category
                                                           : elementwise_node_category,
        broadcast_shape_node_type)
{
    m_opnodes[0] = op;
}

dnd::broadcast_shape_expr_node::broadcast_shape_expr_node(int ndim, const intptr_t *shape,
                                                ndarray_expr_node_ptr&& op)
    : ndarray_expr_node(op->get_dtype(), ndim, 1, shape,
        op->get_node_category() == strided_array_node_category ? strided_array_node_category
                                                           : elementwise_node_category,
        broadcast_shape_node_type)
{
    m_opnodes[0] = std::move(op);
}

ndarray_expr_node_ptr dnd::broadcast_shape_expr_node::apply_linear_index(
                    int ndim, const intptr_t *shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, bool allow_in_place)
{
    ndarray_expr_node *node = m_opnodes[0].get();

    if (allow_in_place) {
        // If the broadcasting doesn't add more dimensions, it's pretty simple
        if (m_ndim == node->get_ndim()) {
            // Adopt the new shape
            m_ndim = ndim;
            memcpy(m_shape.get(), shape, ndim * sizeof(intptr_t));

            m_opnodes[0] = node->apply_linear_index(ndim, shape, axis_map,
                                    index_strides, start_index, m_opnodes[0]->unique());
            return ndarray_expr_node_ptr(this);
        } else {
            throw std::runtime_error("broadcast_shape_expr::apply_linear_index is not completed yet");
        }
    } else {
        throw std::runtime_error("broadcast_shape_expr::apply_linear_index is not completed yet");
    }
}

void dnd::broadcast_shape_expr_node::as_data_and_strides(char **out_originptr,
                                                    intptr_t *out_strides) const
{
    ndarray_expr_node *op = m_opnodes[0].get();
    int dimdelta = get_ndim() - op->get_ndim();

    op->as_data_and_strides(out_originptr, out_strides + dimdelta);
    memset(out_strides, 0, dimdelta * sizeof(intptr_t));
}

// linear_index_expr_node

dnd::linear_index_expr_node::linear_index_expr_node(int ndim, const intptr_t *shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, ndarray_expr_node *op)
    : ndarray_expr_node(op->get_dtype(), ndim, 1, shape,
        op->get_node_category() == strided_array_node_category ? strided_array_node_category
                                                           : arbitrary_node_category,
        linear_index_node_type),
        m_axis_map(ndim), m_index_strides(ndim), m_start_index(op->get_ndim())
{
    m_opnodes[0].reset(op);
    memcpy(m_axis_map.get(), axis_map, ndim * sizeof(int));
    memcpy(m_index_strides.get(), index_strides, ndim * sizeof(intptr_t));
    memcpy(m_start_index.get(), start_index, op->get_ndim() * sizeof(intptr_t));
}

// Since this is a linear_index_expr_node, it absorbs any further linear indexing
ndarray_expr_node_ptr dnd::linear_index_expr_node::apply_linear_index(
                    int ndim, const intptr_t *shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, bool allow_in_place)
{
    ndarray_expr_node *node = m_opnodes[0].get();

    if (allow_in_place) {
        // Adjust the start_index vector 
        for (int i = 0; i < m_ndim; ++i) {
            m_start_index[m_axis_map[i]] += start_index[i];
        }

        // Create the new index_strides
        shortvector<intptr_t> new_index_strides(ndim);
        for (int i = 0; i < ndim; ++i) {
            new_index_strides[i] = index_strides[i] * m_index_strides[axis_map[i]];
        }
        m_index_strides.swap(new_index_strides);

        // Create the new axis_map
        shortvector<int> new_axis_map(ndim);
        for (int i = 0; i < ndim; ++i) {
            new_axis_map[i] = m_axis_map[axis_map[i]];
        }
        m_axis_map.swap(new_axis_map);

        // Copy the shape
        m_ndim = ndim;
        memcpy(m_shape.get(), shape, ndim * sizeof(intptr_t));

        return ndarray_expr_node_ptr(this);
    } else {
        // Create the new start_index
        shortvector<intptr_t> new_start_index(node->get_ndim());
        memcpy(new_start_index.get(), m_start_index.get(), node->get_ndim() * sizeof(intptr_t));
        for (int i = 0; i < m_ndim; ++i) {
            new_start_index[m_axis_map[i]] += start_index[i];
        }

        // Create the new index_strides
        shortvector<intptr_t> new_index_strides(ndim);
        for (int i = 0; i < ndim; ++i) {
            new_index_strides[i] = index_strides[i] * m_index_strides[axis_map[i]];
        }

        // Create the new axis_map
        shortvector<int> new_axis_map(ndim);
        for (int i = 0; i < ndim; ++i) {
            new_axis_map[i] = m_axis_map[axis_map[i]];
        }

        // Create the new node to return
        return ndarray_expr_node_ptr(
                    new linear_index_expr_node(ndim, shape, new_axis_map.get(),
                                new_index_strides.get(), new_start_index.get(), node));
    }
}

void dnd::linear_index_expr_node::debug_dump_extra(ostream& o, const string& indent) const
{
    int op_ndim = m_opnodes[0]->get_ndim();
    shortvector<char> op_useddims(op_ndim);
    memset(op_useddims.get(), 0, op_ndim);

    o << indent << " axis map: (";
    for (int i = 0; i < m_ndim; ++i) {
        o << m_axis_map[i];
        if (i != m_ndim - 1) {
            o << " ";
        }
        op_useddims[i] = 1;
    }
    o << ")\n";
    o << indent << " index strides: (";
    for (int i = 0; i < m_ndim; ++i) {
        o << m_index_strides[i];
        if (i != m_ndim - 1) {
            o << " ";
        }
    }
    o << ")\n";
    o << indent << " start index: (";
    for (int i = 0; i < op_ndim; ++i) {
        // Put the dims which aren't used (i.e. which were eliminated by an integer index)
        // in brackets
        if (op_useddims[i]) {
            o << m_index_strides[i];
        } else {
            o << "[" << m_index_strides[i] << "]";
        }
        if (i != op_ndim - 1) {
            o << " ";
        }
    }
    o << ")\n";
}


// Node factory functions

ndarray_expr_node_ptr dnd::make_strided_array_expr_node(
            const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr,
            const dnd::shared_ptr<void>& buffer_owner)
{
    intptr_t align_bits = (dt.alignment() - 1);
    if (align_bits != 0) {
        intptr_t align_check = reinterpret_cast<intptr_t>(originptr);
        for (int i = 0; i < ndim; ++i) {
            align_check |= strides[i];
        }

        // If the data isn't aligned, return a misbehaved strided node
        if ((align_check & align_bits) != 0) {
            return ndarray_expr_node_ptr(new misbehaved_strided_array_expr_node(dt, ndim,
                                                shape, strides, originptr, buffer_owner));
        }
    }

    // The data type is NBO and the data is aligned, return a strided node
    return ndarray_expr_node_ptr(new strided_array_expr_node(dt, ndim,
                                        shape, strides, originptr, buffer_owner));
}

ndarray_expr_node_ptr dnd::make_convert_dtype_expr_node(
            ndarray_expr_node *node, const dtype& dt, assign_error_mode errmode)
{
    if (node->get_node_type() == strided_array_node_type) {
        return ndarray_expr_node_ptr(new strided_array_expr_node(make_conversion_dtype(dt, node->get_dtype(), errmode),
                    node->get_ndim(), node->get_shape(),
                    static_cast<const strided_array_expr_node *>(node)->get_strides(),
                    static_cast<const strided_array_expr_node *>(node)->get_originptr(),
                    static_cast<const strided_array_expr_node *>(node)->get_buffer_owner()));
    } else {
        // TODO: Remove convert_dtype_expr_node, just use the conversion_dtype
        throw std::runtime_error("can only make a convert<> dtype with strided array nodes presently");
    }
}

ndarray_expr_node_ptr dnd::make_broadcast_strided_array_expr_node(ndarray_expr_node *node,
                                int ndim, const intptr_t *shape,
                                const dtype& dt, assign_error_mode errmode)
{
    if (node->get_node_type() != strided_array_node_type) {
        throw std::runtime_error("broadcasting only supports strided arrays so far");
    }

    strided_array_expr_node *snode = static_cast<strided_array_expr_node *>(node);

    // Broadcast the array's strides to the desired shape (may raise a broadcast error)
    dimvector strides(ndim);
    broadcast_to_shape(ndim, shape, snode, strides.get());

    // Create the strided array node
    ndarray_expr_node_ptr new_node(new strided_array_expr_node(
                    (dt == snode->get_dtype()) ? snode->get_dtype() : make_conversion_dtype(dt, snode->get_dtype(), errmode),
                    ndim, shape, strides.get(), snode->get_originptr(), snode->get_buffer_owner()));

    return std::move(new_node);
}

ndarray_expr_node_ptr dnd::make_linear_index_expr_node(ndarray_expr_node *node,
                                int nindex, const irange *indices, bool allow_in_place)
{
    // Validate the number of indices
    if (nindex > node->get_ndim()) {
        throw too_many_indices(nindex, node->get_ndim());
    }

    // Determine how many dimensions the new array will have
    int new_ndim = node->get_ndim();
    for (int i = 0; i < nindex; ++i) {
        if (indices[i].step() == 0) {
            --new_ndim;
        }
    }

    const intptr_t *shape = node->get_shape();

    // Convert the indices into the form used by linear_index_expr_node
    dimvector new_shape(new_ndim);
    shortvector<int> axis_map(new_ndim);
    shortvector<intptr_t> index_strides(new_ndim), start_index(node->get_ndim());
    int new_i = 0;
    for (int i = 0; i < nindex; ++i) {
        intptr_t step = indices[i].step();
        if (step == 0) {
            // A single index
            intptr_t idx = indices[i].start();
            if (idx < 0 || idx >= shape[i]) {
                throw index_out_of_bounds(idx, 0, shape[i]);
            }
            start_index[i] = idx;
        } else if (step > 0) {
            // A range with a positive step
            intptr_t start = indices[i].start();
            if (start < 0 || start >= shape[i]) {
                if (start == std::numeric_limits<intptr_t>::min()) {
                    start = 0;
                } else {
                    throw irange_out_of_bounds(indices[i], 0, shape[i]);
                }
            }
            start_index[i] = start;

            intptr_t end = indices[i].finish();
            if (end > shape[i]) {
                if (end == std::numeric_limits<intptr_t>::max()) {
                    end = shape[i];
                } else {
                    throw irange_out_of_bounds(indices[i], 0, shape[i]);
                }
            }
            end -= start;
            if (end > 0) {
                if (step == 1) {
                    new_shape[new_i] = end;
                    index_strides[new_i] = 1;
                } else {
                    new_shape[new_i] = (end + step - 1) / step;
                    index_strides[new_i] = step;
                }
            } else {
                new_shape[new_i] = 0;
                index_strides[new_i] = 0;
            }
            axis_map[new_i] = i;
            ++new_i;
        } else {
            // A range with a negative step
            intptr_t start = indices[i].start();
            if (start < 0 || start >= shape[i]) {
                if (start == std::numeric_limits<intptr_t>::min()) {
                    start = shape[i] - 1;
                } else {
                    throw irange_out_of_bounds(indices[i], 0, shape[i]);
                }
            }
            start_index[i] = start;

            intptr_t end = indices[i].finish();
            if (end == std::numeric_limits<intptr_t>::max()) {
                end = -1;
            } else if (end < -1) {
                throw irange_out_of_bounds(indices[i], 0, shape[i]);
            }
            end -= start;
            if (end < 0) {
                if (step == -1) {
                    new_shape[new_i] = -end;
                    index_strides[new_i] = -1;
                } else {
                    new_shape[new_i] = (-end - step - 1) / (-step);
                    index_strides[new_i] = step;
                }
            } else {
                new_shape[new_i] = 0;
                index_strides[new_i] = 0;
            }
            axis_map[new_i] = i;
            ++new_i;
        }
    }
    // Initialize the info for the rest of the dimensions which remain as is
    for (int i = nindex; i < node->get_ndim(); ++i) {
        start_index[i] = 0;
    }
    for (int i = new_i; i < new_ndim; ++i) {
        int old_i = i - new_i + nindex;
        new_shape[i] = shape[old_i];
        index_strides[i] = 1;
        axis_map[i] = old_i;
    }

    /*
    cout << "new_ndim: " << new_ndim << "\n";
    cout << "new_shape: ";
    for (int i = 0; i < new_ndim; ++i) cout << new_shape[i] << " ";
    cout << "\n";
    cout << "axis_map: ";
    for (int i = 0; i < new_ndim; ++i) cout << axis_map[i] << " ";
    cout << "\n";
    cout << "index_strides: ";
    for (int i = 0; i < new_ndim; ++i) cout << index_strides[i] << " ";
    cout << "\n";
    cout << "start_index: ";
    for (int i = 0; i < node->get_ndim(); ++i) cout << start_index[i] << " ";
    cout << "\n";
    */

    return node->apply_linear_index(new_ndim, new_shape.get(), axis_map.get(),
                    index_strides.get(), start_index.get(), allow_in_place);
}

ndarray_expr_node_ptr dnd::make_integer_index_expr_node(ndarray_expr_node *node,
                                int axis, intptr_t idx, bool allow_in_place)
{
    // Validate the axis
    if (axis < 0 || axis >= node->get_ndim()) {
        throw axis_out_of_bounds(axis, 0, node->get_ndim());
    }

    // Validate the index
    if (idx < 0 || idx >= node->get_shape()[axis]) {
        throw index_out_of_bounds(idx, 0, node->get_shape()[axis]);
    }

    return node->apply_integer_index(axis, idx, allow_in_place);
}
