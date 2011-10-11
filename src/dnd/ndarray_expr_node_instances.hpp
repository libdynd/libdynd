//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__NDARRAY_EXPR_NODE_INSTANCES_HPP_
#define _DND__NDARRAY_EXPR_NODE_INSTANCES_HPP_

#include <dnd/ndarray_expr_node.hpp>
#include <dnd/shape_tools.hpp>

namespace dnd {

class ndarray;

/**
 * NDArray expression node which copies the input array as a new data type. As a side
 * effect, it can also be used to align the array data when it is not aligned.
 */
class convert_dtype_expr_node : public ndarray_expr_node {
    assign_error_mode m_errmode;

    convert_dtype_expr_node(const dtype& dt, assign_error_mode errmode, const ndarray_expr_node_ptr& op);
    convert_dtype_expr_node(const dtype& dt, assign_error_mode errmode, ndarray_expr_node_ptr&& op);
public:

    virtual ~convert_dtype_expr_node() {
    }

    ndarray_expr_node_ptr apply_linear_index(int ndim, const intptr_t *shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, bool allow_in_place);

    const char *node_name() const {
        return "convert_dtype";
    }

    void debug_dump_extra(std::ostream& o, const std::string& indent) const;

    friend ndarray_expr_node_ptr make_aligned_expr_node(ndarray_expr_node *node, const dtype& dt,
                                        assign_error_mode errmode);
    friend ndarray_expr_node_ptr make_broadcast_strided_array_expr_node(ndarray_expr_node *node,
                                int ndim, const intptr_t *shape,
                                const dtype& dt, assign_error_mode errmode);
};


template <class BinaryOperatorFactory>
ndarray_expr_node_ptr make_elementwise_binary_op_expr_node(ndarray_expr_node *node1,
                                            ndarray_expr_node *node2, BinaryOperatorFactory& op_factory,
                                            assign_error_mode errmode);

/**
 * NDArray expression node which broadcasts its input to the output.
 */
class broadcast_shape_expr_node : public ndarray_expr_node {
    /**
     * Creates the shape broadcasting node. This function doesn't check
     * that the broadcasting is valid, the caller must validate this
     * before constructing the node.
     */
    broadcast_shape_expr_node(int ndim, const intptr_t *shape, const ndarray_expr_node_ptr& op);
    broadcast_shape_expr_node(int ndim, const intptr_t *shape, ndarray_expr_node_ptr&& op);
public:

    virtual ~broadcast_shape_expr_node() {
    }

    ndarray_expr_node_ptr apply_linear_index(int ndim, const intptr_t *shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, bool allow_in_place);

    /** Provides the data pointer and strides array for the tree evaluation code */
    void as_data_and_strides(char **out_originptr, intptr_t *out_strides) const;

    const char *node_name() const {
        return "broadcast_shape_expr_node";
    }
};

/**
 * NDArray expression node which applies linear indexing.
 *
 * This is an operation core to strided multi-dimensional arrays, so
 * there is also a corresponding function apply_linear_index, which is
 * used to move linear indexing operations as far to the leaves as possible,
 * hopefully absorbing them into the representations of the leaf nodes.
 */
class linear_index_expr_node : public ndarray_expr_node {
    /** For each result axis, gives the corresponding axis in the operand */
    shortvector<int> m_axis_map;
    /** For each result axis, gives the index stride */
    shortvector<intptr_t> m_index_strides;
    /** For each operand axis, gives the start index */
    shortvector<intptr_t> m_start_index;

    /** Creates a linear index node from all the raw components */
    linear_index_expr_node(int ndim, const intptr_t *shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, ndarray_expr_node *op);
public:

    ndarray_expr_node_ptr apply_linear_index(int ndim, const intptr_t *shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, bool allow_in_place);

    const char *node_name() const {
        return "linear_index_expr_node";
    }

    void debug_dump_extra(std::ostream& o, const std::string& indent) const;

    friend ndarray_expr_node_ptr make_linear_index_expr_node(ndarray_expr_node *node,
                                    int nindex, const irange *indices, bool allow_in_place);
    friend class ndarray_expr_node;
};

/**
 * NDArray expression node for element-wise binary operations.
 */
template <class BinaryOperatorFactory>
class elementwise_binary_op_expr_node : public ndarray_expr_node {
    BinaryOperatorFactory m_op_factory;
    /**
     * Constructs the node.
     *
     * IMPORTANT: The input nodes MUST already have been broadcast to identical
     *            shapes and converted to appropriate dtypes for the BinaryOperatorFactory.
     *            These things are not checked by the constructor.
     */
    elementwise_binary_op_expr_node(const ndarray_expr_node_ptr& op0, const ndarray_expr_node_ptr& op1,
                                    BinaryOperatorFactory& op_factory)
            : ndarray_expr_node(op0->get_dtype(), op0->get_ndim(), 2, op0->get_shape(),
                elementwise_node_category, elementwise_binary_op_node_type),
                m_op_factory()
    {
        m_opnodes[0] = op0;
        m_opnodes[1] = op1;

        // Swap in the operator factory
        m_op_factory.swap(op_factory);
    }
    elementwise_binary_op_expr_node(ndarray_expr_node_ptr&& op0, ndarray_expr_node_ptr&& op1,
                                    BinaryOperatorFactory& op_factory)
        : ndarray_expr_node(op0->get_dtype(), op0->get_ndim(), 2, op0->get_shape(),
                elementwise_node_category, elementwise_binary_op_node_type),
                m_op_factory()
    {
        m_opnodes[0] = std::move(op0);
        m_opnodes[1] = std::move(op1);

        // Swap in the operator factory
        m_op_factory.swap(op_factory);
    }

public:

    virtual ~elementwise_binary_op_expr_node() {
    }

    std::pair<binary_operation_t, std::shared_ptr<auxiliary_data> >
            get_binary_operation(intptr_t dst_fixedstride, intptr_t src1_fixedstride,
                                intptr_t src2_fixedstride) const {
        return m_op_factory.get_binary_operation(dst_fixedstride, src1_fixedstride, src2_fixedstride);
    }

    /**
     * Application of a linear index to an elementwise binary operation is propagated to both
     * the input operands.
     */
    ndarray_expr_node_ptr apply_linear_index(int ndim, const intptr_t *shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, bool allow_in_place)
    {
        if (allow_in_place) {
            m_ndim = ndim;
            memcpy(m_shape.get(), shape, ndim * sizeof(intptr_t));
            m_opnodes[0] = m_opnodes[0]->apply_linear_index(ndim, shape, axis_map,
                                            index_strides, start_index, m_opnodes[0]->unique());
            m_opnodes[1] = m_opnodes[1]->apply_linear_index(ndim, shape, axis_map,
                                            index_strides, start_index, m_opnodes[1]->unique());
            return ndarray_expr_node_ptr(this);
        } else {
            ndarray_expr_node_ptr node1, node2;
            node1 = m_opnodes[0]->apply_linear_index(ndim, shape, axis_map,
                                            index_strides, start_index, false);
            node2 = m_opnodes[1]->apply_linear_index(ndim, shape, axis_map,
                                            index_strides, start_index, false);

            BinaryOperatorFactory op_factory_copy(m_op_factory);
            return ndarray_expr_node_ptr(
                        new elementwise_binary_op_expr_node(node1, node2, op_factory_copy));
        }
    }


    const char *node_name() const {
        return m_op_factory.node_name();
    }

    friend ndarray_expr_node_ptr make_elementwise_binary_op_expr_node<BinaryOperatorFactory>(
                                            ndarray_expr_node *node1,
                                            ndarray_expr_node *node2, BinaryOperatorFactory& op_factory,
                                            assign_error_mode errmode);
};

/** Creates an expr node out of the raw data for a strided array */
ndarray_expr_node_ptr make_strided_array_expr_node(
            const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr,
            const std::shared_ptr<void>& buffer_owner);

/**
 * Creates an aligned strided_array_expr_node, possibly with a follow-on node to make
 * the data aligned if it is not.
 *
 * @param a  The array to expose as an expr node.
 * @param dt The data type the node should be. This may cause a dtype conversion
 *           node to be added.
 * @param errmode  The error mode to be used during dtype conversion.
 */
ndarray_expr_node_ptr make_aligned_expr_node(ndarray_expr_node *node, const dtype& dt,
                                        assign_error_mode errmode);

/**
 * Creates an aligned strided_array_expr_node, possibly with a follow-on node to make
 * the data aligned if it is not, baking the shape broadcasting into the strided_array_expr_node.
 *
 * @param node   The strided array node to process.
 * @param ndim   The number of dimensions to broadcast to.
 * @param shape  The shape to broadcast to.
 * @param dt     The data type the node should be. This may cause a dtype conversion
 *               node to be added.
 * @param errmode  The error mode to be used during dtype conversion.
 */
ndarray_expr_node_ptr make_broadcast_strided_array_expr_node(ndarray_expr_node *node,
                                int ndim, const intptr_t *shape,
                                const dtype& dt, assign_error_mode errmode);

/**
 * Applies a linear index to the ndarray node.
 */
ndarray_expr_node_ptr make_linear_index_expr_node(ndarray_expr_node *node,
                                int nindex, const irange *indices, bool allow_in_place);

/**
 * Applies an integer index to the ndarray node.
 */
ndarray_expr_node_ptr make_integer_index_expr_node(ndarray_expr_node *node,
                                int axis, intptr_t idx, bool allow_in_place);


/**
 * Creates an elementwise binary operator node from the two input ndarrays.
 *
 * The contents of op_factory are stolen via a swap() operation.
 */
template<class BinaryOperatorFactory>
ndarray_expr_node_ptr make_elementwise_binary_op_expr_node(ndarray_expr_node *node1,
                                            ndarray_expr_node *node2, BinaryOperatorFactory& op_factory,
                                            assign_error_mode errmode)
{
    // This caches the dtype promotion information in op_factory
    op_factory.promote_dtypes(node1->get_dtype(), node2->get_dtype());

    ndarray_expr_node_ptr broadcast_node1, broadcast_node2;

    // If the shapes match exactly, no need to broadcast.
    if (node1->get_ndim() == node2->get_ndim() &&
                    memcmp(node1->get_shape(), node2->get_shape(), node1->get_ndim() * sizeof(intptr_t)) == 0) {
        broadcast_node1 = make_aligned_expr_node(node1, op_factory.get_dtype(1), errmode);
        broadcast_node2 = make_aligned_expr_node(node2, op_factory.get_dtype(2), errmode);
    } else {
        int op0_ndim;
        dimvector op0_shape;
        broadcast_input_shapes(node1, node2, &op0_ndim, &op0_shape);

        broadcast_node1 = make_broadcast_strided_array_expr_node(node1, op0_ndim,
                                                        op0_shape.get(), op_factory.get_dtype(1), errmode);
        broadcast_node2 = make_broadcast_strided_array_expr_node(node2, op0_ndim,
                                                        op0_shape.get(), op_factory.get_dtype(2), errmode);
    }

    boost::intrusive_ptr<elementwise_binary_op_expr_node<BinaryOperatorFactory> > result(
                new elementwise_binary_op_expr_node<BinaryOperatorFactory>(
                                std::move(broadcast_node1), std::move(broadcast_node2), op_factory));
    
    return result;
}

} // namespace dnd

#endif // _DND__NDARRAY_EXPR_NODE_INSTANCES_HPP_
