//
// Copyright (C) 2011-2012, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/nodes/elementwise_binary_kernel_node.hpp>
#include <dnd/dtypes/convert_dtype.hpp>
#include <dnd/memblock/ndarray_node_memory_block.hpp>
#include <dnd/kernels/assignment_kernels.hpp>
#include <dnd/kernels/buffered_binary_kernels.hpp>

using namespace std;
using namespace dnd;

ndarray_node_ptr dnd::elementwise_binary_kernel_node::as_dtype(const dtype& dt,
                    dnd::assign_error_mode errmode, bool allow_in_place)
{
    if (allow_in_place) {
        m_dtype = make_convert_dtype(dt, m_dtype, errmode);
        return as_ndarray_node_ptr();
    } else {
        ndarray_node_ptr result(
                make_elementwise_binary_kernel_node_copy_kernel(make_convert_dtype(dt, m_dtype, errmode),
                                m_opnodes[0], m_opnodes[1], m_kernel));
        return result;
    }
}

ndarray_node_ptr dnd::elementwise_binary_kernel_node::apply_linear_index(
                int ndim, const bool *remove_axis,
                const intptr_t *start_index, const intptr_t *index_strides,
                const intptr_t *shape,
                bool allow_in_place)
{
    if (allow_in_place) {
        // Apply the indexing to the children
        m_opnodes[0] = m_opnodes[0]->apply_linear_index(ndim, remove_axis,
                                        start_index, index_strides, shape, m_opnodes[0].unique());
        m_opnodes[1] = m_opnodes[1]->apply_linear_index(ndim, remove_axis,
                                        start_index, index_strides, shape, m_opnodes[1].unique());

        broadcast_input_shapes(2, m_opnodes, &m_ndim, &m_shape);

        return as_ndarray_node_ptr();
    } else {
        ndarray_node_ptr node1, node2;
        node1 = m_opnodes[0]->apply_linear_index(ndim, remove_axis,
                                        start_index, index_strides, shape, false);
        node2 = m_opnodes[1]->apply_linear_index(ndim, remove_axis,
                                        start_index, index_strides, shape, false);

        return ndarray_node_ptr(
                    make_elementwise_binary_kernel_node_copy_kernel(m_dtype, node1, node2, m_kernel));
    }
}

void dnd::elementwise_binary_kernel_node::get_binary_operation(intptr_t dst_fixedstride, intptr_t src0_fixedstride,
                            intptr_t src1_fixedstride,
                            kernel_instance<binary_operation_t>& out_kernel) const
{
    if (m_dtype.kind() != expression_kind &&
                        m_opnodes[0]->get_dtype().kind() != expression_kind &&
                        m_opnodes[1]->get_dtype().kind() != expression_kind) {
        // Return the binary operation kernel as is
        out_kernel.kernel = m_kernel.kernel;
        m_kernel.auxdata.borrow_into(out_kernel.auxdata);
    } else {
        // Need to buffer the binary operation kernel.
        kernel_instance<binary_operation_t> kernel;
        unary_specialization_kernel_instance adapters_spec[3];
        kernel_instance<unary_operation_t> adapters[3];
        intptr_t element_sizes[3];

        // Adapt the output
        if (m_dtype.kind() == expression_kind) {
            element_sizes[0] = m_dtype.element_size();
            get_dtype_assignment_kernel(m_dtype.value_dtype(), m_dtype, assign_error_none, adapters_spec[0]);
            adapters[0].kernel = adapters_spec[0].specializations[
                    get_unary_specialization(dst_fixedstride, m_dtype.value_dtype().element_size(),
                                        m_dtype.element_size(), m_dtype.element_size())];
            adapters[0].auxdata.swap(adapters_spec[0].auxdata);
        } else {
            element_sizes[0] = dst_fixedstride;
        }

        // Adapt the first operand
        if (m_opnodes[0]->get_dtype().kind() == expression_kind) {
            element_sizes[1] = m_opnodes[0]->get_dtype().value_dtype().element_size();
            get_dtype_assignment_kernel(m_opnodes[0]->get_dtype().value_dtype(), m_opnodes[0]->get_dtype(),
                                assign_error_none, adapters_spec[1]);
            adapters[1].kernel = adapters_spec[1].specializations[
                        get_unary_specialization(element_sizes[1], element_sizes[1],
                                                src0_fixedstride, m_opnodes[0]->get_dtype().element_size())];
            adapters[1].auxdata.swap(adapters_spec[1].auxdata);
        } else {
            element_sizes[1] = src0_fixedstride;
        }

        // Adapt the second operand
        if (m_opnodes[1]->get_dtype().kind() == expression_kind) {
            element_sizes[2] = m_opnodes[1]->get_dtype().value_dtype().element_size();
            get_dtype_assignment_kernel(m_opnodes[1]->get_dtype().value_dtype(), m_opnodes[1]->get_dtype(),
                                assign_error_none, adapters_spec[2]);
            adapters[2].kernel = adapters_spec[2].specializations[
                        get_unary_specialization(element_sizes[2], element_sizes[2],
                                                src1_fixedstride, m_opnodes[1]->get_dtype().element_size())];
            adapters[2].auxdata.swap(adapters_spec[2].auxdata);
        } else {
            element_sizes[2] = src0_fixedstride;
        }

        // Return the binary operation kernel
        kernel.kernel = m_kernel.kernel;
        m_kernel.auxdata.borrow_into(kernel.auxdata);

        // Hook up the buffering
        make_buffered_binary_kernel(kernel, adapters, element_sizes, out_kernel);
    }
}


ndarray_node_ptr dnd::make_elementwise_binary_kernel_node_copy_kernel(const dtype& dt,
                    const ndarray_node_ptr& opnode0, const ndarray_node_ptr& opnode1,
                    const kernel_instance<binary_operation_t>& kernel)
{
    char *node_memory = NULL;
    ndarray_node_ptr result(make_uninitialized_ndarray_node_memory_block(sizeof(elementwise_binary_kernel_node), &node_memory));

    // Placement new
    new (node_memory) elementwise_binary_kernel_node(
                        dt, opnode0, opnode1, kernel);

    return DND_MOVE(result);
}

ndarray_node_ptr dnd::make_elementwise_binary_kernel_node_steal_kernel(const dtype& dt,
                    const ndarray_node_ptr& opnode0, const ndarray_node_ptr& opnode1,
                    kernel_instance<binary_operation_t>& kernel)
{
    char *node_memory = NULL;
    ndarray_node_ptr result(make_uninitialized_ndarray_node_memory_block(sizeof(elementwise_binary_kernel_node), &node_memory));

    // Placement new
    elementwise_binary_kernel_node *ukn = new (node_memory) elementwise_binary_kernel_node(
                        dt, opnode0, opnode1);

    ukn->m_kernel.swap(kernel);

    return DND_MOVE(result);
}
