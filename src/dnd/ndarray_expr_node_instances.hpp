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

/**
 * NDArray expression node which holds a raw strided array.
 */
class strided_array_expr_node : public ndarray_expr_node {
    char *m_originptr;
    dimvector m_strides;
    std::shared_ptr<void> m_buffer_owner;

    /** Creates a strided array node from the raw values */
    strided_array_expr_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr, const std::shared_ptr<void>& buffer_owner);
public:

    virtual ~strided_array_expr_node() {
    }

    /** Provides the data pointer and strides array for the tree evaluation code */
    void as_data_and_strides(char **out_originptr, intptr_t *out_strides) const;

    const char *node_name() const {
        return "strided_array";
    }

    void debug_dump_extra(std::ostream& o, const std::string& indent) const;

    friend ndarray_expr_node_ptr make_strided_array_expr_node(const ndarray& a);
    friend ndarray_expr_node_ptr make_strided_array_expr_node(const ndarray& a, const dtype& dt,
                                        assign_error_mode errmode);
    friend ndarray_expr_node_ptr make_broadcast_strided_array_expr_node(const ndarray& a,
                                        int ndim, const intptr_t *shape,
                                        const dtype& dt, assign_error_mode errmode);
};

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

    const char *node_name() const {
        return "convert_dtype";
    }

    void debug_dump_extra(std::ostream& o, const std::string& indent) const;

    friend ndarray_expr_node_ptr make_strided_array_expr_node(const ndarray& a);
    friend ndarray_expr_node_ptr make_strided_array_expr_node(const ndarray& a, const dtype& dt,
                                        assign_error_mode);
    friend ndarray_expr_node_ptr make_broadcast_strided_array_expr_node(const ndarray& a,
                                        int ndim, const intptr_t *shape,
                                        const dtype& dt, assign_error_mode errmode);
};


template <class BinaryOperatorFactory>
ndarray_expr_node_ptr make_elementwise_binary_op_expr_node(
                                                        const ndarray& op1, const ndarray& op2,
                                                        BinaryOperatorFactory& op_factory);

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

    /** Provides the data pointer and strides array for the tree evaluation code */
    void as_data_and_strides(char **out_originptr, intptr_t *out_strides) const;

    const char *node_name() const {
        return "broadcast_shape_expr_node";
    }
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
            : ndarray_expr_node(op0->get_dtype(), op0->ndim(), 2, op0->shape(),
                elementwise_node_category, elementwise_binary_op_node_type) {
        m_opnodes[0] = op0;
        m_opnodes[1] = op1;

        // Swap in the operator factory
        m_op_factory.swap(op_factory);
    }
    elementwise_binary_op_expr_node(ndarray_expr_node_ptr&& op0, ndarray_expr_node_ptr&& op1,
                                    BinaryOperatorFactory& op_factory)
            : ndarray_expr_node(op0->get_dtype(), op0->ndim(), 2, op0->shape(),
                elementwise_node_category, elementwise_binary_op_node_type) {
        m_opnodes[0] = std::move(op0);
        m_opnodes[1] = std::move(op1);

        // Swap in the operator factory
        m_op_factory.swap(op_factory);
    }

    std::pair<binary_operation_t, std::shared_ptr<auxiliary_data> >
            get_binary_operation(intptr_t dst_fixedstride, intptr_t src1_fixedstride,
                                intptr_t src2_fixedstride) const {
        return m_op_factory.get_binary_operation(dst_fixedstride, src1_fixedstride, src2_fixedstride);
    }

public:

    virtual ~elementwise_binary_op_expr_node() {
    }

    const char *node_name() const {
        return m_op_factory.node_name();
    }

    friend ndarray_expr_node_ptr make_elementwise_binary_op_expr_node<BinaryOperatorFactory>(
                                                            const ndarray& op1, const ndarray& op2,
                                                            BinaryOperatorFactory& op_factory);
};

/**
 * Creates an aligned strided_array_expr_node, possibly with a follow-on node to make
 * the data aligned if it is not.
 *
 * @param a  The array to expose as an expr node.
 */
ndarray_expr_node_ptr make_strided_array_expr_node(const ndarray& a);

/**
 * Creates an aligned strided_array_expr_node, possibly with a follow-on node to make
 * the data aligned if it is not.
 *
 * @param a  The array to expose as an expr node.
 * @param dt The data type the node should be. This may cause a dtype conversion
 *           node to be added.
 * @param errmode  The error mode to be used during dtype conversion.
 */
ndarray_expr_node_ptr make_strided_array_expr_node(const ndarray& a, const dtype& dt,
                                        assign_error_mode errmode = assign_error_fractional);

/**
 * Creates an aligned strided_array_expr_node, possibly with a follow-on node to make
 * the data aligned if it is not, baking the shape broadcasting into the strided_array_expr_node.
 *
 * @param a      The array to expose as an expr node.
 * @param ndim   The number of dimensions to broadcast to.
 * @param shape  The shape to broadcast to.
 * @param dt     The data type the node should be. This may cause a dtype conversion
 *               node to be added.
 * @param errmode  The error mode to be used during dtype conversion.
 */
ndarray_expr_node_ptr make_broadcast_strided_array_expr_node(const ndarray& a, int ndim, const intptr_t *shape,
                                    const dtype& dt, assign_error_mode errmode = assign_error_fractional);

/**
 * Creates an elementwise binary operator node from the two input ndarrays.
 *
 * The contents of op_factory are stolen via a swap() operation.
 */
template<class BinaryOperatorFactory>
ndarray_expr_node_ptr make_elementwise_binary_op_expr_node(const ndarray& op1, const ndarray& op2,
                                                        BinaryOperatorFactory& op_factory)
{
    if (op1.originptr() == NULL || op2.originptr() == NULL) {
        throw std::runtime_error("cannot yet create an elementwise_binary_op_expr_node from an "
                            "ndarray which is an expression view");
    }

    // This caches the dtype promotion information in op_factory
    op_factory.promote_dtypes(op1.get_dtype(), op2.get_dtype());

    ndarray_expr_node_ptr node1, node2;

    // If the shapes match exactly, no need to broadcast.
    if (op1.ndim() == op2.ndim() && memcmp(op1.shape(), op2.shape(), op1.ndim() * sizeof(intptr_t)) == 0) {
        node1 = make_strided_array_expr_node(op1, op_factory.get_dtype(1));
        node2 = make_strided_array_expr_node(op2, op_factory.get_dtype(2));
    } else {
        int op0_ndim;
        dimvector op0_shape;
        broadcast_input_shapes(op1, op2, &op0_ndim, &op0_shape);

        node1 = make_broadcast_strided_array_expr_node(op1, op0_ndim, op0_shape.get(), op_factory.get_dtype(1));
        node2 = make_broadcast_strided_array_expr_node(op2, op0_ndim, op0_shape.get(), op_factory.get_dtype(2));
    }

    boost::intrusive_ptr<elementwise_binary_op_expr_node<BinaryOperatorFactory> > result(
                new elementwise_binary_op_expr_node<BinaryOperatorFactory>(
                                std::move(node1), std::move(node2), op_factory));
    
    return result;
}

} // namespace dnd

#endif // _DND__NDARRAY_EXPR_NODE_INSTANCES_HPP_
