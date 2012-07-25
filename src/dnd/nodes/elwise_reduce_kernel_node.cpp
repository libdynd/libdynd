//
// Copyright (C) 2011-2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/nodes/elwise_reduce_kernel_node.hpp>
#include <dnd/memblock/ndarray_node_memory_block.hpp>
#include <dnd/nodes/elwise_unary_kernel_node.hpp>
#include <dnd/kernels/assignment_kernels.hpp>
#include <dnd/dtypes/convert_dtype.hpp>

using namespace std;
using namespace dnd;

dnd::elwise_reduce_kernel_node::elwise_reduce_kernel_node(const dtype& dt, const ndarray_node_ptr& opnode, dnd_bool *reduce_axes, bool rightassoc, bool keepdims)
    : m_dtype(dt), m_opnode(opnode), m_kernel(), m_rightassoc(rightassoc), m_keepdims(keepdims),
        m_reduce_axes(opnode->get_ndim(), reduce_axes)
{
    const intptr_t *opnode_shape = opnode->get_shape();
    m_ndim = opnode->get_ndim();
    m_shape.init(m_ndim);
    // Calculate the result shape
    if (m_keepdims) {
        for (int i = 0; i < m_ndim; ++i) {
            m_shape[i] = reduce_axes[i] ? 1 : opnode_shape[i];
        }
    } else {
        int j = 0;
        for (int i = 0; i < m_ndim; ++i) {
            if (!reduce_axes[i]) {
                m_shape[j++] = opnode_shape[i];
            }
        }
        m_ndim = j;
    }
}

ndarray_node_ptr dnd::elwise_reduce_kernel_node::as_dtype(const dtype& dt,
                    dnd::assign_error_mode errmode, bool allow_in_place)
{
    if (allow_in_place) {
        m_dtype = make_convert_dtype(dt, m_dtype, errmode);
        return as_ndarray_node_ptr();
    } else {
        ndarray_node_ptr result(
                make_elwise_reduce_kernel_node_copy_kernel(make_convert_dtype(dt, m_dtype, errmode),
                                m_opnode, m_reduce_axes.get(), m_rightassoc, m_keepdims, m_kernel));
        return result;
    }
}

ndarray_node_ptr dnd::elwise_reduce_kernel_node::apply_linear_index(
                int ndim, const bool *remove_axis,
                const intptr_t *start_index, const intptr_t *index_strides,
                const intptr_t *shape,
                bool allow_in_place)
{
    throw std::runtime_error("TODO: elwise_reduce_kernel_node::apply_linear_index");
}

ndarray_node_ptr dnd::make_elwise_reduce_kernel_node_copy_kernel(const dtype& dt, const ndarray_node_ptr& opnode,
                                            dnd_bool *reduce_axes, bool rightassoc, bool keepdims,
                                            const kernel_instance<unary_operation_t>& kernel)
{
    char *node_memory = NULL;
    ndarray_node_ptr result(make_uninitialized_ndarray_node_memory_block(sizeof(elwise_reduce_kernel_node), &node_memory));

    // Placement new
    elwise_reduce_kernel_node *ukn = new (node_memory) elwise_reduce_kernel_node(
                        dt, opnode, reduce_axes, rightassoc, keepdims);

    ukn->m_kernel.copy_from(kernel);

    return DND_MOVE(result);
}

ndarray_node_ptr dnd::make_elwise_reduce_kernel_node_steal_kernel(const dtype& dt, const ndarray_node_ptr& opnode,
                                            dnd_bool *reduce_axes, bool rightassoc, bool keepdims,
                                            kernel_instance<unary_operation_t>& kernel)
{
    char *node_memory = NULL;
    ndarray_node_ptr result(make_uninitialized_ndarray_node_memory_block(sizeof(elwise_reduce_kernel_node), &node_memory));

    // Placement new
    elwise_reduce_kernel_node *ukn = new (node_memory) elwise_reduce_kernel_node(
                        dt, opnode, reduce_axes, rightassoc, keepdims);

    ukn->m_kernel.swap(kernel);

    return DND_MOVE(result);
}
