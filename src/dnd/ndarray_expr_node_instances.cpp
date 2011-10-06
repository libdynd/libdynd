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

void dnd::broadcast_shape_expr_node::as_data_and_strides(char **out_originptr,
                                                    intptr_t *out_strides) const
{
    const ndarray_expr_node *op = m_opnodes[0].get();
    int dimdelta = ndim() - op->ndim();

    op->as_data_and_strides(out_originptr, out_strides + dimdelta);
    memset(out_strides, 0, dimdelta * sizeof(intptr_t));
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

