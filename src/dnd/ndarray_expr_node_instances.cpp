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

#include "ndarray_expr_node_instances.hpp"

using namespace std;
using namespace dnd;

// strided_array_expr_node

dnd::strided_array_expr_node::strided_array_expr_node(const dtype& dt, int ndim,
                                const intptr_t *shape, const intptr_t *strides,
                                char *originptr, const std::shared_ptr<void>& buffer_owner)
    : ndarray_expr_node(dt, ndim, 0, shape,
        strided_array_node_category, strided_array_node_type),
      m_originptr(originptr), m_strides(ndim, strides),
      m_buffer_owner(buffer_owner)
{
}


void dnd::strided_array_expr_node::as_data_and_strides(char **out_originptr,
                                                    intptr_t *out_strides) const
{
    *out_originptr = m_originptr;
    memcpy(out_strides, m_strides.get(), ndim() * sizeof(intptr_t));
}

ndarray_expr_node_ptr dnd::strided_array_expr_node::apply_linear_index(
                    int new_ndim, const intptr_t *new_shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, bool allow_in_place)
{
    if (allow_in_place) {
        // Apply the start_index to m_originptr
        for (int i = 0; i < m_ndim; ++i) {
            m_originptr += m_strides[i] * start_index[i];
        }

        // Adopt the new shape
        m_ndim = new_ndim;
        memcpy(m_shape.get(), new_shape, new_ndim * sizeof(intptr_t));

        // Construct the new strides
        dimvector new_strides(m_ndim);
        for (int i = 0; i < new_ndim; ++i) {
            new_strides[i] = m_strides[axis_map[i]] * index_strides[i];
        }
        m_strides.swap(new_strides);

        return ndarray_expr_node_ptr(this);
    } else {
        // Apply the start_index to m_originptr
        char *new_originptr = m_originptr;
        for (int i = 0; i < m_ndim; ++i) {
            new_originptr += m_strides[i] * start_index[i];
        }

        // Construct the new strides
        dimvector new_strides(m_ndim);
        for (int i = 0; i < new_ndim; ++i) {
            new_strides[i] = m_strides[axis_map[i]] * index_strides[i];
        }

        return ndarray_expr_node_ptr(
            new strided_array_expr_node(m_dtype, new_ndim, new_shape, new_strides.get(),
                                        new_originptr, m_buffer_owner));
    }
}

void dnd::strided_array_expr_node::debug_dump_extra(ostream& o, const string& indent) const
{
    o << indent << " strides: ";
    for (int i = 0; i < m_ndim; ++i) {
        o << m_strides[i] << " ";
    }
    o << "\n";
    o << indent << " originptr: " << (void *)m_originptr << "\n";
    o << indent << " buffer owner: " << m_buffer_owner.get() << "\n";
}

// convert_dtype_expr_node

dnd::convert_dtype_expr_node::convert_dtype_expr_node(const dtype& dt,
                                    assign_error_mode errmode, const ndarray_expr_node_ptr& op)
    : ndarray_expr_node(dt, op->ndim(), 1, op->shape(), elementwise_node_category,
        convert_dtype_node_type), m_errmode(errmode)
{
    m_opnodes[0] = op;
}

dnd::convert_dtype_expr_node::convert_dtype_expr_node(const dtype& dt,
                                    assign_error_mode errmode, ndarray_expr_node_ptr&& op)
    : ndarray_expr_node(dt, op->ndim(), 1, op->shape(), elementwise_node_category,
        convert_dtype_node_type), m_errmode(errmode)
{
    m_opnodes[0] = std::move(op);
}

void dnd::convert_dtype_expr_node::debug_dump_extra(ostream& o, const string& indent) const
{
    o << indent << " error mode: ";
    switch (m_errmode) {
        case assign_error_none:
            o << "assign_error_none";
            break;
        case assign_error_overflow:
            o << "assign_error_overflow";
            break;
        case assign_error_fractional:
            o << "assign_error_fractional";
            break;
        case assign_error_inexact:
            o << "assign_error_inexact";
            break;
        default:
            o << "unknown error mode (" << (int)m_errmode << ")";
            break;
    }
    o << "\n";
}

ndarray_expr_node_ptr dnd::convert_dtype_expr_node::apply_linear_index(
                    int ndim, const intptr_t *shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, bool allow_in_place)
{
    ndarray_expr_node *node = m_opnodes[0].get();

    if (allow_in_place) {
        // Adopt the new shape
        m_ndim = ndim;
        memcpy(m_shape.get(), shape, ndim * sizeof(intptr_t));

        m_opnodes[0] = node->apply_linear_index(ndim, shape, axis_map,
                                index_strides, start_index, m_opnodes[0]->unique());
        return ndarray_expr_node_ptr(this);
    } else {
        return ndarray_expr_node_ptr(
            new convert_dtype_expr_node(m_dtype, m_errmode,
                                node->apply_linear_index(ndim, shape, axis_map,
                                    index_strides, start_index, false)));
    }
}


// broadcast_shape_expr_node

dnd::broadcast_shape_expr_node::broadcast_shape_expr_node(int ndim, const intptr_t *shape,
                                                const ndarray_expr_node_ptr& op)
    : ndarray_expr_node(op->get_dtype(), ndim, 1, shape,
        op->node_category() == strided_array_node_category ? strided_array_node_category
                                                           : elementwise_node_category,
        broadcast_shape_node_type)
{
    m_opnodes[0] = op;
}

dnd::broadcast_shape_expr_node::broadcast_shape_expr_node(int ndim, const intptr_t *shape,
                                                ndarray_expr_node_ptr&& op)
    : ndarray_expr_node(op->get_dtype(), ndim, 1, shape,
        op->node_category() == strided_array_node_category ? strided_array_node_category
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
        if (m_ndim == node->ndim()) {
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
    int dimdelta = ndim() - op->ndim();

    op->as_data_and_strides(out_originptr, out_strides + dimdelta);
    memset(out_strides, 0, dimdelta * sizeof(intptr_t));
}

// linear_index_expr_node

dnd::linear_index_expr_node::linear_index_expr_node(int ndim, const intptr_t *shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, ndarray_expr_node *op)
    : ndarray_expr_node(op->get_dtype(), ndim, 1, shape,
        op->node_category() == strided_array_node_category ? strided_array_node_category
                                                           : arbitrary_node_category,
        linear_index_node_type), m_axis_map(ndim), m_index_strides(ndim), m_start_index(op->ndim())
{
    m_opnodes[0].reset(op);
    memcpy(m_axis_map.get(), axis_map, ndim * sizeof(int));
    memcpy(m_index_strides.get(), index_strides, ndim * sizeof(intptr_t));
    memcpy(m_start_index.get(), start_index, op->ndim() * sizeof(intptr_t));
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
        shortvector<intptr_t> new_start_index(node->ndim());
        memcpy(new_start_index.get(), m_start_index.get(), node->ndim() * sizeof(intptr_t));
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

// Node factory functions

ndarray_expr_node_ptr dnd::make_strided_array_expr_node(const ndarray& a)
{
    if (a.originptr() == NULL) {
        throw std::runtime_error("cannot create a strided_array_expr_node from an "
                            "ndarray which is an expression view");
    }

    // Create the strided array node
    ndarray_expr_node_ptr a_node(new strided_array_expr_node(a.get_dtype(), a.ndim(),
                                        a.shape(), a.strides(), a.originptr(), a.buffer_owner()));

    // Add an alignment node if necessary (the convert_dtype node with equal dtype
    // will cause alignment)
    if (a.is_aligned()) {
        return std::move(a_node);
    } else {
        return ndarray_expr_node_ptr(
                new convert_dtype_expr_node(a.get_dtype(), assign_error_none, std::move(a_node)));
    }
}

ndarray_expr_node_ptr dnd::make_strided_array_expr_node(const ndarray& a, const dtype& dt,
                                        assign_error_mode errmode)
{
    if (a.originptr() == NULL) {
        throw std::runtime_error("cannot create a strided_array_expr_node from an "
                            "ndarray which is an expression view");
    }

    // Create the strided array node
    ndarray_expr_node_ptr a_node(new strided_array_expr_node(a.get_dtype(), a.ndim(),
                                        a.shape(), a.strides(), a.originptr(), a.buffer_owner()));

    // Add an conversion/alignment node if necessary
    if (dt == a.get_dtype() && a.is_aligned()) {
        return std::move(a_node);
    } else {
        return ndarray_expr_node_ptr(
                new convert_dtype_expr_node(dt, errmode, std::move(a_node)));
    }
}

ndarray_expr_node_ptr dnd::make_broadcast_strided_array_expr_node(const ndarray& a,
                                int ndim, const intptr_t *shape,
                                const dtype& dt, assign_error_mode errmode)
{
    if (a.originptr() == NULL) {
        throw std::runtime_error("cannot create a strided_array_expr_node from an "
                            "ndarray which is an expression view");
    }

    // Broadcast the array's strides to the desired shape (may raise a broadcast error)
    dimvector strides(ndim);
    broadcast_to_shape(ndim, shape, a, strides.get());

    // Create the strided array node
    ndarray_expr_node_ptr a_node(new strided_array_expr_node(a.get_dtype(), ndim,
                                        shape, strides.get(), a.originptr(), a.buffer_owner()));

    // Add an conversion/alignment node if necessary
    if (dt == a.get_dtype() && a.is_aligned()) {
        return std::move(a_node);
    } else {
        return ndarray_expr_node_ptr(
                new convert_dtype_expr_node(dt, errmode, std::move(a_node)));
    }
}

ndarray_expr_node_ptr dnd::make_linear_index_expr_node(ndarray_expr_node *node,
                                int nindex, const irange *indices, bool allow_in_place)
{
    // Validate the number of indices
    if (nindex > node->ndim()) {
        throw too_many_indices(nindex, node->ndim());
    }

    // Determine how many dimensions the new array will have
    int new_ndim = node->ndim();
    for (int i = 0; i < nindex; ++i) {
        if (indices[i].step() == 0) {
            --new_ndim;
        }
    }

    const intptr_t *shape = node->shape();

    // Convert the indices into the form used by linear_index_expr_node
    dimvector new_shape(new_ndim);
    shortvector<int> axis_map(new_ndim);
    shortvector<intptr_t> index_strides(new_ndim), start_index(node->ndim());
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
                if (start == INTPTR_MIN) {
                    start = 0;
                } else {
                    throw irange_out_of_bounds(indices[i], 0, shape[i]);
                }
            }
            start_index[i] = start;

            intptr_t end = indices[i].finish();
            if (end > shape[i]) {
                if (end == INTPTR_MAX) {
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
                if (start == INTPTR_MIN) {
                    start = shape[i] - 1;
                } else {
                    throw irange_out_of_bounds(indices[i], 0, shape[i]);
                }
            }
            start_index[i] = start;

            intptr_t end = indices[i].finish();
            if (end == INTPTR_MAX) {
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
                    index_strides[new_i] = -step;
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
    for (int i = nindex; i < node->ndim(); ++i) {
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
    for (int i = 0; i < node->ndim(); ++i) cout << start_index[i] << " ";
    cout << "\n";
    */

    return node->apply_linear_index(new_ndim, new_shape.get(), axis_map.get(),
                    index_strides.get(), start_index.get(), allow_in_place);
}
