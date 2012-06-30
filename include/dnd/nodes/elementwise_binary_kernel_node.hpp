//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__ELEMENTWISE_BINARY_KERNEL_NODE_HPP_
#define _DND__ELEMENTWISE_BINARY_KERNEL_NODE_HPP_

#include <dnd/nodes/ndarray_node.hpp>
#include <dnd/shape_tools.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>
#include <dnd/kernels/buffered_binary_kernels.hpp>
#include <dnd/kernels/assignment_kernels.hpp>

namespace dnd {

class ndarray;

template<class BinaryOperatorFactory>
ndarray_node_ptr make_elementwise_binary_kernel_node(ndarray_node_ptr node1,
                                            ndarray_node_ptr node2, BinaryOperatorFactory& op_factory,
                                            assign_error_mode errmode);

template<class BinaryOperatorFactory>
ndarray_node_ptr make_elementwise_binary_kernel_node(const dtype& dt, int ndim, const intptr_t *shape,
                        const ndarray_node_ptr& op0, const ndarray_node_ptr& op1,
                        BinaryOperatorFactory& op_factory);

/**
 * NDArray expression node for element-wise binary operations.
 */
template <class BinaryOperatorFactory>
class elementwise_binary_kernel_node : public ndarray_node {
    /* The number of dimensions in the result array */
    int m_ndim;
    /* The shape of the result array */
    dimvector m_shape;
    /* The data type of this node's result */
    dtype m_dtype;
    /* Pointers to the child nodes */
    ndarray_node_ptr m_opnodes[2];
    BinaryOperatorFactory m_op_factory;
    /**
     * Constructs the node.
     */
    elementwise_binary_kernel_node(const dtype& dt, int ndim, const intptr_t *shape,
                        const ndarray_node_ptr& op0, const ndarray_node_ptr& op1,
                        BinaryOperatorFactory& op_factory)
        : m_ndim(ndim), m_shape(ndim, shape), m_dtype(dt), m_op_factory()
    {
        m_opnodes[0] = op0;
        m_opnodes[1] = op1;

        // Swap in the operator factory
        m_op_factory.swap(op_factory);
    }

public:

    virtual ~elementwise_binary_kernel_node() {
    }

    ndarray_node_category get_category() const
    {
        return elementwise_node_category;
    }

    const dtype& get_dtype() const {
        return m_dtype;
    }

    int get_ndim() const {
        return m_ndim;
    }

    const intptr_t *get_shape() const
    {
        return m_shape.get();
    }

    uint32_t get_access_flags() const
    {
        // Readable, and inherit the immutable access flag of the operands
        return read_access_flag |
            (m_opnodes[0]->get_access_flags() & m_opnodes[1]->get_access_flags() & immutable_access_flag);
    }
    
    int get_nop() const {
        return 2;
    }

    const ndarray_node_ptr& get_opnode(int i) const {
        return m_opnodes[i];
    }

    memory_block_ptr get_memory_block() const
    {
        return memory_block_ptr();
    }

    ndarray_node_ptr as_dtype(const dtype& dt,
                        dnd::assign_error_mode errmode, bool allow_in_place)
    {
        if (allow_in_place) {
            m_dtype = make_conversion_dtype(dt, m_dtype, errmode);
            return as_ndarray_node_ptr();
        } else {
            ndarray_node_ptr result(
                    make_elementwise_binary_kernel_node(make_conversion_dtype(dt, m_dtype, errmode),
                                    m_ndim, m_shape.get(), m_opnodes[0], m_opnodes[1], m_op_factory));
            return result;
        }
    }

    void get_binary_operation(intptr_t dst_fixedstride, intptr_t src0_fixedstride,
                                intptr_t src1_fixedstride,
                                kernel_instance<binary_operation_t>& out_kernel) const
    {
        if (m_dtype.kind() != expression_kind &&
                            m_opnodes[0]->get_dtype().kind() != expression_kind &&
                            m_opnodes[1]->get_dtype().kind() != expression_kind) {
            // Return the binary operation kernel as is
            m_op_factory.get_binary_operation(dst_fixedstride, src0_fixedstride, src1_fixedstride, out_kernel);
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

            // Get the binary operation kernel with the adapted strides
            m_op_factory.get_binary_operation(element_sizes[0], element_sizes[1], element_sizes[2], kernel);

            // Hook up the buffering
            make_buffered_binary_kernel(kernel, adapters, element_sizes, out_kernel);
        }
    }

    /**
     * Application of a linear index to an elementwise binary operation is propagated to both
     * the input operands.
     */
    ndarray_node_ptr apply_linear_index(
                    int ndim, const bool *remove_axis,
                    const intptr_t *start_index, const intptr_t *index_strides,
                    const intptr_t *shape,
                    bool allow_in_place)
    {
        if (allow_in_place) {
            // Apply the indexing to the childrin
            m_opnodes[0] = m_opnodes[0]->apply_linear_index(ndim, remove_axis,
                                            start_index, index_strides, shape, m_opnodes[0].unique());
            m_opnodes[1] = m_opnodes[1]->apply_linear_index(ndim, remove_axis,
                                            start_index, index_strides, shape, m_opnodes[1].unique());

            // Broadcast the new child shapes to get this node's shape
            broadcast_input_shapes(2, m_opnodes, &m_ndim, &m_shape);

            return as_ndarray_node_ptr();
        } else {
            ndarray_node_ptr node1, node2;
            node1 = m_opnodes[0]->apply_linear_index(ndim, remove_axis,
                                            start_index, index_strides, shape, false);
            node2 = m_opnodes[1]->apply_linear_index(ndim, remove_axis,
                                            start_index, index_strides, shape, false);

            // Broadcast the new child shapes to get the new node's shape
            int new_ndim;
            dimvector new_shape;
            broadcast_input_shapes(node1, node2, &new_ndim, &new_shape);

            BinaryOperatorFactory op_factory_copy(m_op_factory);
            return ndarray_node_ptr(
                        make_elementwise_binary_kernel_node(m_dtype, new_ndim, new_shape.get(), node1, node2, op_factory_copy));
        }
    }


    const char *node_name() const {
        return m_op_factory.node_name();
    }

    friend ndarray_node_ptr make_elementwise_binary_kernel_node<BinaryOperatorFactory>(
                                            ndarray_node_ptr node1,
                                            ndarray_node_ptr node2, BinaryOperatorFactory& op_factory,
                                            assign_error_mode errmode);

    friend ndarray_node_ptr make_elementwise_binary_kernel_node<BinaryOperatorFactory>(const dtype& dt, int ndim, const intptr_t *shape,
                            const ndarray_node_ptr& op0, const ndarray_node_ptr& op1,
                            BinaryOperatorFactory& op_factory);
};

/**
 * Creates an elementwise binary operator node from the two input ndarrays.
 *
 * The contents of op_factory are stolen via a swap() operation.
 */
template<class BinaryOperatorFactory>
ndarray_node_ptr make_elementwise_binary_kernel_node(ndarray_node_ptr node1,
                                            ndarray_node_ptr node2, BinaryOperatorFactory& op_factory,
                                            assign_error_mode errmode)
{
    // op_factory caches the dtype promotion information
    op_factory.promote_dtypes(node1->get_dtype(), node2->get_dtype());

    // Determine which dtypes need conversion
    if (node1->get_dtype() != op_factory.get_dtype(1)) {
        node1 = node1->as_dtype(op_factory.get_dtype(1), errmode, false);
    }
    if (node2->get_dtype() != op_factory.get_dtype(2)) {
        node2 = node2->as_dtype(op_factory.get_dtype(2), errmode, false);
    }

    // Allocate the memory_block
    char *result = reinterpret_cast<char *>(malloc(sizeof(memory_block_data) + sizeof(elementwise_binary_kernel_node<BinaryOperatorFactory>)));
    if (result == NULL) {
        throw bad_alloc();
    }

    // If the shapes match exactly, no need to broadcast.
    if (node1->get_ndim() == node2->get_ndim() &&
                    memcmp(node1->get_shape(), node2->get_shape(),
                            node1->get_ndim() * sizeof(intptr_t)) == 0) {
        // Placement new
        new (result + sizeof(memory_block_data)) elementwise_binary_kernel_node<BinaryOperatorFactory>(
                            op_factory.get_dtype(0), node1->get_ndim(), node1->get_shape(),
                            node1, node2, op_factory);
    } else {
        int op0_ndim;
        dimvector op0_shape;
        broadcast_input_shapes(node1, node2, &op0_ndim, &op0_shape);

        // Placement new
        new (result + sizeof(memory_block_data)) elementwise_binary_kernel_node<BinaryOperatorFactory>(
                            op_factory.get_dtype(0), op0_ndim, op0_shape.get(),
                            node1, node2, op_factory);
    }

    return ndarray_node_ptr(new (result) memory_block_data(1, ndarray_node_memory_block_type), false);
}

template<class BinaryOperatorFactory>
ndarray_node_ptr make_elementwise_binary_kernel_node(const dtype& dt, int ndim, const intptr_t *shape,
                        const ndarray_node_ptr& op0, const ndarray_node_ptr& op1,
                        BinaryOperatorFactory& op_factory)
{
    // Allocate the memory_block
    char *result = reinterpret_cast<char *>(malloc(sizeof(memory_block_data) + sizeof(elementwise_binary_kernel_node<BinaryOperatorFactory>)));
    if (result == NULL) {
        throw bad_alloc();
    }

    // Placement new
    new (result + sizeof(memory_block_data)) elementwise_binary_kernel_node<BinaryOperatorFactory>(
                        dt, ndim, shape,
                        op0, op1, op_factory);

    return ndarray_node_ptr(new (result) memory_block_data(1, ndarray_node_memory_block_type), false);
}

} // namespace dnd

#endif // _DND__ELEMENTWISE_BINARY_KERNEL_NODE_HPP_
