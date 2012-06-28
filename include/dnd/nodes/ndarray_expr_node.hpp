//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__NDARRAY_EXPR_NODE_HPP_
#define _DND__NDARRAY_EXPR_NODE_HPP_

#include <boost/detail/atomic_count.hpp>
#include <boost/intrusive_ptr.hpp>

#include <dnd/dtype.hpp>
#include <dnd/dtype_assign.hpp>
#include <dnd/irange.hpp>
#include <dnd/kernels/kernel_instance.hpp>
#include <dnd/shortvector.hpp>
#include <dnd/memory_block.hpp>

namespace dnd {

class ndarray;

enum expr_node_category {
    // The node points a simple strided array in memory
    strided_array_node_category,
    // The node represents an elementwise nop() to 1 transformation
    elementwise_node_category,
    // The node represents an arbitrary computation node, which will generally
    // require evaluation to a temporary.
    arbitrary_node_category
};

enum expr_node_type {
    // This node represents an NBO, aligned, strided array
    strided_array_node_type,
    // This node represents a single scalar data element, by value
    immutable_scalar_node_type,
    broadcast_shape_node_type,
    elementwise_binary_op_node_type,
    linear_index_node_type
};


/**
 * Virtual base class for the ndarray expression tree.
 *
 * TODO: Model this after how LLVM does this kind of thing?
 */
class ndarray_expr_node {
#ifdef DND_CLING
    // A hack avoiding boost atomic_count, since that creates inline assembly which LLVM JIT doesn't like!
    mutable long m_use_count;
#else
    /** Embedded reference counting using boost::intrusive_ptr */
    mutable boost::detail::atomic_count m_use_count;
#endif

    // Non-copyable
    ndarray_expr_node(const ndarray_expr_node&);
    ndarray_expr_node& operator=(const ndarray_expr_node&);

protected:
    expr_node_type m_node_type;
    expr_node_category m_node_category;
    /* The data type of this node's result */
    dtype m_dtype;
    /* The number of dimensions in the result array */
    int m_ndim;
    /* The number of child operands this node uses */
    int m_nop;
    /* The shape of the result array */
    dimvector m_shape;
    /* Pointers to the child nodes */
    shortvector<boost::intrusive_ptr<ndarray_expr_node> > m_opnodes;

    /**
     * Constructs the basic node with NULL operand children.
     */
    ndarray_expr_node(const dtype& dt, int ndim, int nop, const intptr_t *shape,
                        expr_node_category node_category, expr_node_type node_type)
        : m_use_count(0), m_node_type(node_type), m_node_category(node_category),
            m_dtype(dt), m_ndim(ndim), m_nop(nop), m_shape(ndim, shape),
            m_opnodes(nop) {
    }

public:
    bool unique() const {
        // If a single intrusive_ptr has been created, the use count will
        // be one. If a raw pointer is being used, the use count will be zero.
        return m_use_count <= 1;
    }

    virtual ~ndarray_expr_node() {
    }

    const dtype& get_dtype() const {
        return m_dtype;
    }

    int get_ndim() const {
        return m_ndim;
    }

    int get_nop() const {
        return m_nop;
    }

    boost::intrusive_ptr<ndarray_expr_node> get_opnode(int i)
    {
        if (i >= 0 && i < m_nop) {
            return m_opnodes[i];
        } else {
            std::stringstream ss;
            ss << "tried to get ndarray_expr_node operand " << i << " from a " << m_nop << "-ary node";
            throw std::runtime_error(ss.str());

        }
    }

    expr_node_category get_node_category() const {
        return m_node_category;
    }

    expr_node_type get_node_type() const {
        return m_node_type;
    }

    const intptr_t *get_shape() const {
        return m_shape.get();
    }

    /**
     * Nodes with the category strided_array_node_category and with writeable data
     * should override this function.
     *
     * This function should push the strides to the right, as the default broadcasting
     * rules.
     *
     * The default implementation raises an exception.
     */
    virtual void as_readwrite_data_and_strides(int ndim, char **out_originptr, intptr_t *out_strides) const;

    /**
     * Nodes with the category strided_array_node_category and with readable
     * data should override this function.
     *
     * This function should push the strides to the right, as the default broadcasting
     * rules.
     *
     * The default implementation raises an exception.
     */
    virtual void as_readonly_data_and_strides(int ndim, char const **out_originptr, intptr_t *out_strides) const;

    virtual void get_nullary_operation(intptr_t dst_fixedstride,
                                    kernel_instance<nullary_operation_t>& out_kernel) const;
    virtual void get_unary_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride,
                                    kernel_instance<unary_operation_t>& out_kernel) const;
    virtual void get_binary_operation(intptr_t dst_fixedstride, intptr_t src1_fixedstride,
                                    intptr_t src2_fixedstride,
                                    kernel_instance<binary_operation_t>& out_kernel) const;

    /**
     * Evaluates the node into a strided array with a dtype that is
     * not expression_kind.
     */
    boost::intrusive_ptr<ndarray_expr_node>  evaluate();

    /**
     * Converts this node to a new dtype. This uses a conversion_dtype.
     */
    virtual boost::intrusive_ptr<ndarray_expr_node> as_dtype(const dtype& dt,
                        assign_error_mode errmode, bool allow_in_place) = 0;

    /**
     * Applies a linear index to the node, returning either the current node (for do-nothing
     * indexes), or a new node with the index applied. This may apply the indexing up
     * the tree, or in cases where this is not possible, return a node which applies the
     * indexing during evaluation.
     *
     * The idea of this operation is that ndim and the shape for which the linear index
     * operation is generated may be larger than that of the node where it gets applied.
     * The requirement is that the node be broadcastable to the operation's shape, something
     * which the caller must be certain of when calling this function. The resulting node
     * is broadcastable to the operation's shape with the invariant that
     * broadcast(linear_index(node)) is equivalent to linear_index(broadcast(node)).
     *
     * IMPORTANT: The input ndim, etc. are not validated, the caller must ensure that
     *            the shape of the node is broadcastable to the shape for which the linear
     *            index is applied.
     */
    virtual boost::intrusive_ptr<ndarray_expr_node> apply_linear_index(
                    int ndim, const bool *remove_axis,
                    const intptr_t *start_index, const intptr_t *index_strides,
                    const intptr_t *shape,
                    bool allow_in_place) = 0;

    /** Debug printing of the tree */
    void debug_dump(std::ostream& o, const std::string& indent) const;
    /** Debug printing of the data from the derived class */
    virtual void debug_dump_extra(std::ostream& o, const std::string& indent) const;

    virtual const char *node_name() const = 0;

    friend void intrusive_ptr_add_ref(const ndarray_expr_node *node);
    friend void intrusive_ptr_release(const ndarray_expr_node *node);
};

/**
 * Use boost::intrusive_ptr as the smart pointer implementation.
 */
typedef boost::intrusive_ptr<ndarray_expr_node> ndarray_expr_node_ptr;

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

} // namespace dnd

#endif // _DND__NDARRAY_EXPR_NODE_HPP_
