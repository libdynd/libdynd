//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__NDARRAY_EXPR_NODE_INSTANCES_HPP_
#define _DND__NDARRAY_EXPR_NODE_INSTANCES_HPP_

#include <dnd/nodes/ndarray_node.hpp>
#include <dnd/shape_tools.hpp>
#include <dnd/dtypes/conversion_dtype.hpp>
#include <dnd/kernels/buffered_binary_kernels.hpp>
#include <dnd/kernels/assignment_kernels.hpp>

namespace dnd {

class ndarray;


template <class BinaryOperatorFactory>
ndarray_node_ref make_elementwise_binary_op_expr_node(ndarray_node *node1,
                                            ndarray_node *node2, BinaryOperatorFactory& op_factory,
                                            assign_error_mode errmode);

/**
 * NDArray expression node for element-wise binary operations.
 */
template <class BinaryOperatorFactory>
class elementwise_binary_op_expr_node : public ndarray_node {
    BinaryOperatorFactory m_op_factory;
    /**
     * Constructs the node.
     *
     * IMPORTANT: The input nodes MUST already have been broadcast to identical
     *            shapes and converted to appropriate dtypes for the BinaryOperatorFactory.
     *            These things are not checked by the constructor.
     */
    elementwise_binary_op_expr_node(const dtype& dt, int ndim, const intptr_t *shape,
                        const ndarray_node_ref& op0, const ndarray_node_ref& op1,
                        BinaryOperatorFactory& op_factory)
        : ndarray_node(dt, ndim, 2, shape,
                elementwise_node_category, elementwise_binary_op_node_type),
                m_op_factory()
    {
        m_opnodes[0] = op0;
        m_opnodes[1] = op1;

        // Swap in the operator factory
        m_op_factory.swap(op_factory);
    }

public:

    virtual ~elementwise_binary_op_expr_node() {
    }

    ndarray_node_ref as_dtype(const dtype& dt,
                        dnd::assign_error_mode errmode, bool allow_in_place)
    {
        if (allow_in_place) {
            m_dtype = make_conversion_dtype(dt, m_dtype, errmode);
            return ndarray_node_ref(this);
        } else {
            ndarray_node_ref result(
                    new elementwise_binary_op_expr_node(make_conversion_dtype(dt, m_dtype, errmode),
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
    ndarray_node_ref apply_linear_index(
                    int ndim, const bool *remove_axis,
                    const intptr_t *start_index, const intptr_t *index_strides,
                    const intptr_t *shape,
                    bool allow_in_place)
    {
        if (allow_in_place) {
            // Apply the indexing to the childrin
            m_opnodes[0] = m_opnodes[0]->apply_linear_index(ndim, remove_axis,
                                            start_index, index_strides, shape, m_opnodes[0]->unique());
            m_opnodes[1] = m_opnodes[1]->apply_linear_index(ndim, remove_axis,
                                            start_index, index_strides, shape, m_opnodes[1]->unique());

            // Broadcast the new child shapes to get this node's shape
            broadcast_input_shapes(m_opnodes[0].get(), m_opnodes[1].get(), &m_ndim, &m_shape);

            return ndarray_node_ref(this);
        } else {
            ndarray_node_ref node1, node2;
            node1 = m_opnodes[0]->apply_linear_index(ndim, remove_axis,
                                            start_index, index_strides, shape, false);
            node2 = m_opnodes[1]->apply_linear_index(ndim, remove_axis,
                                            start_index, index_strides, shape, false);

            // Broadcast the new child shapes to get the new node's shape
            int new_ndim;
            dimvector new_shape;
            broadcast_input_shapes(node1.get(), node2.get(), &new_ndim, &new_shape);

            BinaryOperatorFactory op_factory_copy(m_op_factory);
            return ndarray_node_ref(
                        new elementwise_binary_op_expr_node(m_dtype, new_ndim, new_shape.get(), node1, node2, op_factory_copy));
        }
    }


    const char *node_name() const {
        return m_op_factory.node_name();
    }

    friend ndarray_node_ref make_elementwise_binary_op_expr_node<BinaryOperatorFactory>(
                                            ndarray_node *node1,
                                            ndarray_node *node2, BinaryOperatorFactory& op_factory,
                                            assign_error_mode errmode);
};

/**
 * Creates an elementwise binary operator node from the two input ndarrays.
 *
 * The contents of op_factory are stolen via a swap() operation.
 */
template<class BinaryOperatorFactory>
ndarray_node_ref make_elementwise_binary_op_expr_node(ndarray_node *node1,
                                            ndarray_node *node2, BinaryOperatorFactory& op_factory,
                                            assign_error_mode errmode)
{
    // op_factory caches the dtype promotion information
    op_factory.promote_dtypes(node1->get_dtype(), node2->get_dtype());

    ndarray_node_ref final_node1, final_node2;

    // Determine which dtypes need conversion
    if (node1->get_dtype() == op_factory.get_dtype(1)) {
        final_node1 = node1;
    } else {
        final_node1 = node1->as_dtype(op_factory.get_dtype(1), errmode, false);
    }
    if (node2->get_dtype() == op_factory.get_dtype(2)) {
        final_node2 = node2;
    } else {
        final_node2 = node2->as_dtype(op_factory.get_dtype(2), errmode, false);
    }

        // If the shapes match exactly, no need to broadcast.
    if (node1->get_ndim() == node2->get_ndim() &&
                    memcmp(node1->get_shape(), node2->get_shape(),
                            node1->get_ndim() * sizeof(intptr_t)) == 0) {
        return ndarray_node_ref(new elementwise_binary_op_expr_node<BinaryOperatorFactory>(
                            op_factory.get_dtype(0), node1->get_ndim(), node1->get_shape(), final_node1, final_node2, op_factory));
    } else {
        int op0_ndim;
        dimvector op0_shape;
        broadcast_input_shapes(final_node1.get(), final_node2.get(), &op0_ndim, &op0_shape);

        return ndarray_node_ref(new elementwise_binary_op_expr_node<BinaryOperatorFactory>(
                            op_factory.get_dtype(0), op0_ndim, op0_shape.get(),
                            final_node1, final_node2, op_factory));
    }
}

} // namespace dnd

#endif // _DND__NDARRAY_EXPR_NODE_INSTANCES_HPP_
