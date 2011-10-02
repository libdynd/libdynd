//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//

#include <stdexcept>
#include <sstream>

#include <dnd/ndarray.hpp>
#include <dnd/ndarray_expr_node.hpp>

using namespace std;
using namespace dnd;

void dnd::ndarray_expr_node::as_data_and_strides(char ** /*out_data*/,
                                                intptr_t * /*out_strides*/) const
{
    throw std::runtime_error("as_data_and_strides is only valid for "
                             "nodes with an expr_node_strided_array category");
}

dnd::strided_array_expr_node::strided_array_expr_node(ndarray& a)
    : ndarray_expr_node(a.get_dtype(), a.ndim(), 0, a.shape(),
        strided_array_node_category, strided_array_node_type),
        m_originptr(a.originptr()), m_strides(a.ndim(), a.strides()),
        m_buffer_owner(a.buffer_owner())
{
    if (m_originptr == NULL) {
        throw std::runtime_error("cannot create a strided_array_expr_node from an "
                            "ndarray which is an expression view");
    }
}

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

void dnd::broadcast_shape_expr_node::as_data_and_strides(char **out_originptr,
                                                    intptr_t *out_strides) const
{
    const ndarray_expr_node *op = m_opnodes[0];
    int dimdelta = ndim() - op->ndim();

    op->as_data_and_strides(out_originptr, out_strides + dimdelta);
    memset(out_strides, 0, dimdelta * sizeof(intptr_t));
}
