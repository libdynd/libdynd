//
// Copyright (C) 2011-2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/nodes/elwise_unary_kernel_node.hpp>
#include <dnd/dtypes/convert_dtype.hpp>
#include <dnd/memblock/ndarray_node_memory_block.hpp>

using namespace std;
using namespace dnd;

ndarray_node_ptr dnd::elwise_unary_kernel_node::as_dtype(const dtype& dt,
                    dnd::assign_error_mode errmode, bool allow_in_place)
{
    if (allow_in_place) {
        m_dtype = make_convert_dtype(dt, m_dtype, errmode);
        return as_ndarray_node_ptr();
    } else {
        ndarray_node_ptr result(
                make_elwise_unary_kernel_node_copy_kernel(make_convert_dtype(dt, m_dtype, errmode),
                                m_opnode, m_kernel));
        return result;
    }
}

ndarray_node_ptr dnd::elwise_unary_kernel_node::apply_linear_index(
                int ndim, const bool *remove_axis,
                const intptr_t *start_index, const intptr_t *index_strides,
                const intptr_t *shape,
                bool allow_in_place)
{
    if (allow_in_place) {
        // Apply the indexing to the children
        m_opnode = m_opnode->apply_linear_index(ndim, remove_axis,
                                        start_index, index_strides, shape, m_opnode.unique());

        return as_ndarray_node_ptr();
    } else {
        ndarray_node_ptr node;
        node = m_opnode->apply_linear_index(ndim, remove_axis,
                                        start_index, index_strides, shape, false);

        return ndarray_node_ptr(
                    make_elwise_unary_kernel_node_copy_kernel(m_dtype, m_opnode, m_kernel));
    }
}

void dnd::elwise_unary_kernel_node::get_unary_specialization_operation(unary_specialization_kernel_instance& out_kernel) const
{
    out_kernel.borrow_from(m_kernel);
}


ndarray_node_ptr dnd::make_elwise_unary_kernel_node_copy_kernel(const dtype& dt, const ndarray_node_ptr& opnode,
                                            const unary_specialization_kernel_instance& kernel)
{
    char *node_memory = NULL;
    ndarray_node_ptr result(make_uninitialized_ndarray_node_memory_block(sizeof(elwise_unary_kernel_node), &node_memory));

    // Placement new
    new (node_memory) elwise_unary_kernel_node(
                        dt, opnode, kernel);

    return DND_MOVE(result);
}

ndarray_node_ptr dnd::make_elwise_unary_kernel_node_steal_kernel(const dtype& dt, const ndarray_node_ptr& opnode,
                                            unary_specialization_kernel_instance& kernel)
{
    char *node_memory = NULL;
    ndarray_node_ptr result(make_uninitialized_ndarray_node_memory_block(sizeof(elwise_unary_kernel_node), &node_memory));

    // Placement new
    elwise_unary_kernel_node *ukn = new (node_memory) elwise_unary_kernel_node(
                        dt, opnode);

    ukn->m_kernel.swap(kernel);

    return DND_MOVE(result);
}
