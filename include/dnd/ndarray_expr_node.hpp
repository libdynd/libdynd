//
// Copyright (C) 2011 Mark Wiebe (mwwiebe@gmail.com)
// All rights reserved.
//
// This is unreleased proprietary software.
//
#ifndef _DND__NDARRAY_EXPR_NODE_HPP_
#define _DND__NDARRAY_EXPR_NODE_HPP_

#include <boost/detail/atomic_count.hpp>
#include <boost/intrusive_ptr.hpp>

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
    convert_dtype_node_type,
    broadcast_shape_node_type
};


/**
 * Virtual base class for the ndarray expression tree.
 *
 * TODO: Model this after how LLVM does this kind of thing?
 */
class ndarray_expr_node {
    mutable boost::detail::atomic_count m_use_count;

    // Non-copyable
    ndarray_expr_node(const ndarray_expr_node&);
    ndarray_expr_node& operator=(const ndarray_expr_node&);

protected:
    /* The data type of this node's result */
    dtype m_dtype;
    /* The number of dimensions in the result array */
    int m_ndim;
    /* The number of child operands this node uses */
    int m_nop;
    /* The shape of the result array */
    dimvector m_shape;
    expr_node_category m_node_category;
    expr_node_type m_node_type;
    /* Pointers to the child nodes */
    shortvector<boost::intrusive_ptr<ndarray_expr_node> > m_opnodes;

    /**
     * Constructs the basic node with NULL operand children.
     */
    ndarray_expr_node(const dtype& dt, int ndim, int nop, const intptr_t *shape,
                        expr_node_category node_category, expr_node_type node_type)
        : m_use_count(0), m_dtype(dt), m_ndim(ndim), m_nop(nop), m_shape(ndim, shape),
            m_node_category(node_category), m_node_type(node_type), m_opnodes(nop) {
    }

public:
    virtual ~ndarray_expr_node() {
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

    expr_node_category node_category() const {
        return m_node_category;
    }

    expr_node_type node_type() const {
        return m_node_type;
    }

    const intptr_t *shape() const {
        return m_shape.get();
    }

    /**
     * Nodes with the category strided_array_node_cateogry should override this function,
     * and fill the outputs.
     *
     * The default implementation raises an exception.
     */
    virtual void as_data_and_strides(char **out_originptr, intptr_t *out_strides) const;

    friend void intrusive_ptr_add_ref(const ndarray_expr_node *node);
    friend void intrusive_ptr_release(const ndarray_expr_node *node);
};

/** Adds a reference, for intrusive_ptr<ndarray_expr_node> to use */
inline void intrusive_ptr_add_ref(const ndarray_expr_node *node) {
    ++node->m_use_count;
}

/** Frees a reference, for intrusive_ptr<ndarray_expr_node> to use */
inline void intrusive_ptr_release(const ndarray_expr_node *node) {
    if (--node->m_use_count == 0) {
        delete node;
    }
}

/**
 * Use boost::intrusive_ptr as the smart pointer implementation.
 */
typedef boost::intrusive_ptr<ndarray_expr_node> ndarray_expr_node_ptr;

/**
 * Creates an aligned strided_array_expr_node, possibly with a follow-on node to make
 * the data aligned if it is not.
 *
 * @param a  The array to expose as an expr node.
 */
ndarray_expr_node_ptr make_strided_array_expr_node(ndarray& a);

/**
 * Creates an aligned strided_array_expr_node, possibly with a follow-on node to make
 * the data aligned if it is not.
 *
 * @param a  The array to expose as an expr node.
 * @param dt The data type the node should be. This may cause a dtype conversion
 *           node to be added.
 * @param errmode  The error mode to be used during dtype conversion.
 */
ndarray_expr_node_ptr make_strided_array_expr_node(ndarray& a, const dtype& dt,
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
ndarray_expr_node_ptr make_broadcast_strided_array_expr_node(ndarray& a, int ndim, const intptr_t *shape,
                                    const dtype& dt, assign_error_mode errmode = assign_error_fractional);


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

    friend ndarray_expr_node_ptr make_strided_array_expr_node(ndarray& a);
    friend ndarray_expr_node_ptr make_strided_array_expr_node(ndarray& a, const dtype& dt,
                                        assign_error_mode errmode);
    friend ndarray_expr_node_ptr make_broadcast_strided_array_expr_node(ndarray& a,
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

    friend ndarray_expr_node_ptr make_strided_array_expr_node(ndarray& a);
    friend ndarray_expr_node_ptr make_strided_array_expr_node(ndarray& a, const dtype& dt,
                                        assign_error_mode);
    friend ndarray_expr_node_ptr make_broadcast_strided_array_expr_node(ndarray& a,
                                        int ndim, const intptr_t *shape,
                                        const dtype& dt, assign_error_mode errmode);
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
    broadcast_shape_expr_node(int ndim, intptr_t *shape, const ndarray_expr_node_ptr& op);
    broadcast_shape_expr_node(int ndim, intptr_t *shape, ndarray_expr_node_ptr&& op);

    virtual ~broadcast_shape_expr_node() {
    }

    /** Provides the data pointer and strides array for the tree evaluation code */
    void as_data_and_strides(char **out_originptr, intptr_t *out_strides) const;
};

} // namespace dnd

#endif // _DND__NDARRAY_EXPR_NODE_HPP_
