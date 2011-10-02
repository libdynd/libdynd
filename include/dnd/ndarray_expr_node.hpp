//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__NDARRAY_EXPR_NODE_HPP_
#define _DND__NDARRAY_EXPR_NODE_HPP_

#include <dnd/dtype.hpp>
#include <dnd/shortvector.hpp>

namespace dnd {

class ndarray;

enum expr_node_category {
    // The node points a simple strided array in memory
    strided_array_node_category,
    // The node represents an elementwise nop() to 1 transformation
    elementwise_node_category
};

enum expr_node_type {
    strided_array_node_type,
    broadcast_shape_node_type
};

/**
 * Virtual base class for the ndarray expression tree.
 *
 * TODO: Model this after how LLVM does this kind of thing?
 */
class ndarray_expr_node {
protected:
    /* The data type of this node's result */
    dtype m_dtype;
    /* The number of dimensions in the result array */
    int m_ndim;
    /* The number of child operands this node uses */
    int m_nop;
    /* The shape of the result array */
    dimvector m_shape;
    expr_node_category m_category;
    expr_node_type m_type;
    /* Pointers to the child nodes */
    shortvector<ndarray_expr_node *> m_opnodes;
public:
    /**
     * Constructs the basic node with NULL operand children.
     */
    ndarray_expr_node(const dtype& dt, int ndim, int nop, const intptr_t *shape,
                        expr_node_category category, expr_node_type type)
        : m_dtype(dt), m_ndim(ndim), m_nop(nop), m_shape(ndim, shape),
            m_category(category), m_type(type), m_opnodes(nop) {
        for (int i = 0; i < m_nop; ++i) {
            m_opnodes[i] = NULL;
        }
    }

    virtual ~ndarray_expr_node() {
        for (int i = 0; i < m_nop; ++i) {
            delete m_opnodes[i];
        }
    }

    const dtype& get_dtype() const {
        return m_dtype;
    }

    int ndim() const {
        return m_ndim;
    }

    int nop() const {
        return m_nop;
    }

    expr_node_category category() const {
        return m_category;
    }

    const intptr_t *shape() const {
        return m_shape.get();
    }

    /**
     * Nodes with the category strided_array_node_cateogry should override this function,
     * and fill the outputs. The default implementation raises an exception.
     */
    virtual void as_data_and_strides(char **out_originptr, intptr_t *out_strides) const;

    /**
     * When the flag ndarray_expr_strided is set, this function
     * may be called to fill in nop() input operand strides, all
     * of which have a shape matching that of this node.
     *
     * Any stride where the corresponding shape is 1 MUST be set
     * to the value 0.
     *
     * @param out_data     The origin data pointers to fill.
     * @param out_strides  The strides to fill.
     * @param axis_perm    NULL for F-order, or the permutation
     *                     representing the order of the axes during
     *                     iteration.
     */
    virtual void fill_strided_ops(char **data, intptr_t **out_strides,
                                    const int *axis_perm) const = 0;
};

/**
 * NDArray expression node which holds a raw strided array.
 */
class strided_array_expr_node : public ndarray_expr_node {
    char *m_originptr;
    dimvector m_strides;
    std::shared_ptr<void> m_buffer_owner;
public:
    /** Creates a strided array node from an ndarray (must have strided array data) */
    strided_array_expr_node(ndarray& a);
    /** Creates a strided array node from the raw values */
    strided_array_expr_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr, const std::shared_ptr<void>& buffer_owner);

    /** Provides the data pointer and strides array for the tree evaluation code */
    void as_data_and_strides(char **out_originptr, intptr_t *out_strides) const;
};

/**
 * NDArray expression node which broadcasts its input to the output.
 *
 * This node takes ownership of the operand node it is given.
 */
class broadcast_shape_expr_node : public ndarray_expr_node {
public:
    /**
     * Creates the shape broadcasting node. This function doesn't check
     * that the broadcasting is valid, the caller must validate this
     * before constructing the node.
     */
    broadcast_shape_expr_node(int ndim, intptr_t *shape, ndarray_expr_node *op)
            : ndarray_expr_node(op->get_dtype(), ndim, 1, shape,
                op->category(), broadcast_shape_node_type) {
        m_opnodes[0] = op;
    }

    virtual ~broadcast_shape_expr_node() {
    }

    /** Provides the data pointer and strides array for the tree evaluation code */
    void as_data_and_strides(char **out_originptr, intptr_t *out_strides) const;
};

} // namespace dnd

#endif // _DND__NDARRAY_EXPR_NODE_HPP_
