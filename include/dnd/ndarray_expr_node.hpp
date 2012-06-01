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
#include <dnd/irange.hpp>
#include <dnd/operations.hpp>
#include <dnd/shortvector.hpp>

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
    // This node represents a strided array which is either not NBO, or is misaligned,
    // i.e. requires buffering
    misbehaved_strided_array_node_type,
    convert_dtype_node_type,
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
     * Nodes with the category strided_array_node_category should override this function,
     * and fill the outputs.
     *
     * The default implementation raises an exception.
     */
    virtual void as_data_and_strides(char **out_originptr, intptr_t *out_strides) const;

    virtual std::pair<nullary_operation_t, dnd::shared_ptr<auxiliary_data> >
                get_nullary_operation(intptr_t dst_fixedstride) const;
    virtual std::pair<unary_operation_t, dnd::shared_ptr<auxiliary_data> >
                get_unary_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride) const;
    virtual std::pair<binary_operation_t, dnd::shared_ptr<auxiliary_data> >
                get_binary_operation(intptr_t dst_fixedstride, intptr_t src1_fixedstride,
                                      intptr_t src2_fixedstride) const;

    /**
     * Evaluates the expression tree into a strided array
     */
    boost::intrusive_ptr<ndarray_expr_node>  evaluate() const;

    /**
     * Applies a linear index to the node, returning either the current node (for do-nothing
     * indexes), or a new node with the index applied. This may apply the indexing up
     * the tree, or in cases where this is not possible, returning a node which applies the
     * indexing.
     *
     * IMPORTANT: The input ndim, shape, etc are not validated, their correctness must
     *            be ensured by the caller.
     */
    virtual boost::intrusive_ptr<ndarray_expr_node> apply_linear_index(
                    int ndim, const intptr_t *shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, bool allow_in_place);
    /**
     * Applies a single integer index to the node. See apply_linear_index for more details.
     *
     * IMPORTANT: The input axis and idx are not validated, their correctness must
     *            be ensured by the caller.
     *
     * @param axis  Which axis to index
     * @param idx   The index to apply
     * @param allow_in_place  The operation is permitted mutate the node data and return 'this'.
     */
    virtual boost::intrusive_ptr<ndarray_expr_node> apply_integer_index(
                                        int axis, intptr_t idx, bool allow_in_place);


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

/**
 * NDArray expression node which holds a raw strided array.
 */
class strided_array_expr_node : public ndarray_expr_node {
    char *m_originptr;
    dimvector m_strides;
    dnd::shared_ptr<void> m_buffer_owner;

    // Non-copyable
    strided_array_expr_node(const strided_array_expr_node&);
    strided_array_expr_node& operator=(const strided_array_expr_node&);

public:
    /**
     * Creates a strided array node from the raw values.
     *
     * The dtype must be NBO, and the data must all be aligned, but this
     * constructor does not validate these constraints. Failure to enforce
     * these contraints will result in undefined behavior.
     *
     * It's prefereable to use the function make_strided_array_expr_node function,
     * as it does the parameter validation.
     */
    strided_array_expr_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr, const dnd::shared_ptr<void>& buffer_owner);

    /**
     * Constructs a strided array node with the given dtype, shape, and axis_perm (for memory layout)
     */
    strided_array_expr_node(const dtype& dt, int ndim, const intptr_t *shape, const int *axis_perm);

    virtual ~strided_array_expr_node() {
    }

    char *get_originptr() const {
        return m_originptr;
    }

    const intptr_t *get_strides() const {
        return m_strides.get();
    }

    dnd::shared_ptr<void> get_buffer_owner() const {
        return m_buffer_owner;
    }

    /** Provides the data pointer and strides array for the tree evaluation code */
    void as_data_and_strides(char **out_originptr, intptr_t *out_strides) const;

    ndarray_expr_node_ptr apply_linear_index(int ndim, const intptr_t *shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, bool allow_in_place);
    // TODO: Implement apply_integer_index

    const char *node_name() const {
        return "strided_array";
    }

    void debug_dump_extra(std::ostream& o, const std::string& indent) const;

    friend class ndarray;
    friend ndarray_expr_node_ptr make_strided_array_expr_node(
                    const dtype& dt, int ndim, const intptr_t *shape,
                    const intptr_t *strides, char *originptr,
                    const dnd::shared_ptr<void>& buffer_owner);
    // TODO: Add a virtual broadcast function to the base node type, then remove this friend function
    friend ndarray_expr_node_ptr make_broadcast_strided_array_expr_node(ndarray_expr_node *node,
                                int ndim, const intptr_t *shape,
                                const dtype& dt, assign_error_mode errmode);
};

/**
 * NDArray expression node which holds a raw strided array that is misaligned.
 */
class misbehaved_strided_array_expr_node : public ndarray_expr_node {
    dtype m_inner_dtype;
    char *m_originptr;
    dimvector m_strides;
    dnd::shared_ptr<void> m_buffer_owner;

    // Non-copyable
    misbehaved_strided_array_expr_node(const misbehaved_strided_array_expr_node&);
    misbehaved_strided_array_expr_node& operator=(const misbehaved_strided_array_expr_node&);

    /** Creates a strided array node from the raw values */
    misbehaved_strided_array_expr_node(const dtype& dt, int ndim, const intptr_t *shape,
            const intptr_t *strides, char *originptr, const dnd::shared_ptr<void>& buffer_owner);

public:

    virtual ~misbehaved_strided_array_expr_node() {
    }

    /** Returns the dtype of the actual binary data */
    const dtype& get_inner_dtype() const {
        return m_inner_dtype;
    }

    char *get_originptr() const {
        return m_originptr;
    }

    const intptr_t *get_strides() const {
        return m_strides.get();
    }

    dnd::shared_ptr<void> get_buffer_owner() const {
        return m_buffer_owner;
    }

    /** Returns true if every element is aligned */
    bool is_aligned() const {
        int alignment = get_dtype().alignment();
        if (alignment == 1) {
            return true;
        } else {
            const intptr_t *strides = get_strides();
            int ndim = get_ndim();
            int align_test = static_cast<int>(reinterpret_cast<intptr_t>(get_originptr()));

            for (int i = 0; i < ndim; ++i) {
                align_test |= static_cast<int>(strides[i]);
            }
            return ((alignment - 1) & align_test) == 0;
        }
    }

    /**
     * Provides the data pointer and strides array for the tree evaluation code. Note that
     * the data may be in a different byte-order, so use the dtype returned by get_inner_dtype().
     * The data may also be misaligned, so don't use operations requiring alignment.
     */
    void as_data_and_strides(char **out_originptr, intptr_t *out_strides) const;

    ndarray_expr_node_ptr apply_linear_index(int ndim, const intptr_t *shape, const int *axis_map,
                    const intptr_t *index_strides, const intptr_t *start_index, bool allow_in_place);
    // TODO: Implement apply_integer_index

    const char *node_name() const {
        return "misbehaved_strided_array";
    }

    void debug_dump_extra(std::ostream& o, const std::string& indent) const;

    friend ndarray_expr_node_ptr make_strided_array_expr_node(
                    const dtype& dt, int ndim, const intptr_t *shape,
                    const intptr_t *strides, char *originptr,
                    const dnd::shared_ptr<void>& buffer_owner);
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

} // namespace dnd

#endif // _DND__NDARRAY_EXPR_NODE_HPP_
