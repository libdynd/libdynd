//
// Copyright (C) 2011-2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/nodes/elwise_reduce_kernel_node.hpp>
#include <dynd/memblock/ndarray_node_memory_block.hpp>
#include <dynd/nodes/elwise_unary_kernel_node.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/dtypes/convert_dtype.hpp>

using namespace std;
using namespace dynd;

dynd::elwise_reduce_kernel_node::elwise_reduce_kernel_node(const dtype& dt,
                        const ndarray_node_ptr& opnode, dnd_bool *reduce_axes, bool rightassoc, bool keepdims, const ndarray_node_ptr& identity)
    : m_dtype(dt), m_opnode(opnode), m_kernel(), m_rightassoc(rightassoc), m_keepdims(keepdims),
        m_identity(identity), m_reduce_axes(opnode->get_ndim(), reduce_axes)
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

ndarray_node_ptr dynd::elwise_reduce_kernel_node::as_dtype(const dtype& dt,
                    dynd::assign_error_mode errmode, bool allow_in_place)
{
    if (m_dtype == dt) {
        return as_ndarray_node_ptr();
    } else if (allow_in_place) {
        m_dtype = make_convert_dtype(dt, m_dtype, errmode);
        return as_ndarray_node_ptr();
    } else {
        ndarray_node_ptr result(
                make_elwise_reduce_kernel_node_copy_kernel(make_convert_dtype(dt, m_dtype, errmode),
                                m_opnode, m_reduce_axes.get(), m_rightassoc, m_keepdims, m_identity, m_kernel));
        return result;
    }
}

ndarray_node_ptr dynd::elwise_reduce_kernel_node::apply_linear_index(
                int DND_UNUSED(ndim), const bool *DND_UNUSED(remove_axis),
                const intptr_t *DND_UNUSED(start_index), const intptr_t *DND_UNUSED(index_strides),
                const intptr_t *DND_UNUSED(shape),
                bool DND_UNUSED(allow_in_place))
{
    throw std::runtime_error("TODO: elwise_reduce_kernel_node::apply_linear_index");
}

void dynd::elwise_reduce_kernel_node::get_unary_operation(intptr_t DND_UNUSED(dst_fixedstride), intptr_t DND_UNUSED(src_fixedstride),
                                    kernel_instance<unary_operation_t>& out_kernel) const
{
    out_kernel.borrow_from(m_kernel);
}

void dynd::elwise_reduce_kernel_node::debug_dump_extra(std::ostream& o, const std::string& indent) const
{
    o << indent << " associative: " << (m_rightassoc ? "right" : "left") << "\n";
    o << indent << " keepdims: " << (m_keepdims ? "true" : "false") << "\n";
    o << indent << " reduce axes: ";
    for (int i = 0, i_end = m_opnode->get_ndim(); i != i_end; ++i) {
        if (m_reduce_axes[i]) {
            o << i << " ";
        }
    }
    o << "\n";
    if (m_identity.get()) {
        o << indent << " reduction identity:\n";
        m_identity->debug_dump(o, indent + " ");
    } else {
        o << indent << " reduction identity: NULL\n";
    }
}

ndarray_node_ptr dynd::make_elwise_reduce_kernel_node_copy_kernel(const dtype& dt, const ndarray_node_ptr& opnode,
                                            dnd_bool *reduce_axes, bool rightassoc, bool keepdims, const ndarray_node_ptr& identity,
                                            const kernel_instance<unary_operation_t>& kernel)
{
    char *node_memory = NULL;
    ndarray_node_ptr result(make_uninitialized_ndarray_node_memory_block(sizeof(elwise_reduce_kernel_node), &node_memory));

    // Placement new
    elwise_reduce_kernel_node *ukn = new (node_memory) elwise_reduce_kernel_node(
                        dt, opnode, reduce_axes, rightassoc, keepdims, identity);

    ukn->m_kernel.copy_from(kernel);

    return DND_MOVE(result);
}

ndarray_node_ptr dynd::make_elwise_reduce_kernel_node_steal_kernel(const dtype& dt, const ndarray_node_ptr& opnode,
                                            dnd_bool *reduce_axes, bool rightassoc, bool keepdims, const ndarray_node_ptr& identity,
                                            kernel_instance<unary_operation_t>& kernel)
{
    char *node_memory = NULL;
    ndarray_node_ptr result(make_uninitialized_ndarray_node_memory_block(sizeof(elwise_reduce_kernel_node), &node_memory));

    // Placement new
    elwise_reduce_kernel_node *ukn = new (node_memory) elwise_reduce_kernel_node(
                        dt, opnode, reduce_axes, rightassoc, keepdims, identity);

    ukn->m_kernel.swap(kernel);

    return DND_MOVE(result);
}
