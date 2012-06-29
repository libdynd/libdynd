//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__NDARRAY_NODE_HPP_
#define _DND__NDARRAY_NODE_HPP_

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

enum ndarray_access_flags {
    /** If an ndarray node is readable */
    read_access_flag = 0x01,
    /** If an ndarray node is writeable */
    write_access_flag = 0x02,
    /** If an ndarray node will not be written to by anyone else either */
    immutable_access_flag = 0x04
};

enum ndarray_node_category {
    // The node points a simple strided array in memory
    strided_array_node_category,
    // The node represents an elementwise nop() to 1 transformation
    elementwise_node_category,
    // The node represents an arbitrary computation node, which will generally
    // require evaluation to a temporary.
    arbitrary_node_category
};

/**
 * Virtual base class for the ndarray expression tree.
 *
 * TODO: Model this after how LLVM does this kind of thing?
 */
class ndarray_node {
#ifdef DND_CLING
    // A hack avoiding boost atomic_count, since that creates inline assembly which LLVM JIT doesn't like!
    mutable long m_use_count;
#else
    /** Embedded reference counting using boost::intrusive_ptr */
    mutable boost::detail::atomic_count m_use_count;
#endif

    // Non-copyable
    ndarray_node(const ndarray_node&);
    ndarray_node& operator=(const ndarray_node&);

protected:

    /**
     * Constructs the basic node with NULL operand children.
     */
    ndarray_node()
        : m_use_count(0)
    {
    }

public:
    bool unique() const {
        // If a single intrusive_ptr has been created, the use count will
        // be one. If a raw pointer is being used, the use count will be zero.
        return m_use_count <= 1;
    }

    virtual ~ndarray_node() {
    }

    virtual ndarray_node_category get_category() const = 0;

    virtual const dtype& get_dtype() const = 0;

    virtual uint32_t get_access_flags() const = 0;

    virtual int get_ndim() const = 0;

    virtual const intptr_t *get_shape() const = 0;

    virtual const intptr_t *get_strides() const;

    /**
     * Gets the strides of this node, with the axes broadcast to the right.
     *
     * IMPORTANT: The caller must validate that the broadcasting is valid, this
     *            function does not do that checking, and can't because it doesn't know
     *            the shape it's being requested for.
     */
    void get_right_broadcast_strides(int ndim, intptr_t *out_strides) const;

    virtual const char *get_readonly_originptr() const;

    virtual char *get_readwrite_originptr() const;

    /**
     * Retrieves the memory_block object which holds the data for this
     * node. If this node holds its own data, returns NULL.
     */
    virtual memory_block_ref get_memory_block() const = 0;

    /** The number of operand nodes this node depends on */
    virtual int get_nop() const
    {
        return 0;
    }

    virtual ndarray_node* get_opnode(int DND_UNUSED(i)) const {
        throw std::runtime_error("This ndarray_node does not have any operand nodes");
    }

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
    boost::intrusive_ptr<ndarray_node>  evaluate();

    /**
     * Converts this node to a new dtype. This uses a conversion_dtype.
     */
    virtual boost::intrusive_ptr<ndarray_node> as_dtype(const dtype& dt,
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
    virtual boost::intrusive_ptr<ndarray_node> apply_linear_index(
                    int ndim, const bool *remove_axis,
                    const intptr_t *start_index, const intptr_t *index_strides,
                    const intptr_t *shape,
                    bool allow_in_place) = 0;

    /** Debug printing of the tree */
    void debug_dump(std::ostream& o, const std::string& indent) const;
    /** Debug printing of the data from the derived class */
    virtual void debug_dump_extra(std::ostream& o, const std::string& indent) const;

    virtual const char *node_name() const = 0;

    friend void intrusive_ptr_add_ref(const ndarray_node *node);
    friend void intrusive_ptr_release(const ndarray_node *node);
};

/**
 * Use boost::intrusive_ptr as the smart pointer implementation.
 */
typedef boost::intrusive_ptr<ndarray_node> ndarray_node_ref;

/** Adds a reference, for intrusive_ptr<ndarray_node> to use */
inline void intrusive_ptr_add_ref(const ndarray_node *node) {
    ++node->m_use_count;
}

/** Frees a reference, for intrusive_ptr<ndarray_node> to use */
inline void intrusive_ptr_release(const ndarray_node *node) {
    if (--node->m_use_count == 0) {
        delete node;
    }
}

/** Applies the slicing index to the ndarray node. */
ndarray_node_ref apply_index_to_node(ndarray_node *node,
                                int nindex, const irange *indices, bool allow_in_place);
/**
 * Applies an integer index to the ndarray node.
 */
ndarray_node_ref apply_integer_index_to_node(ndarray_node *node,
                                int axis, intptr_t idx, bool allow_in_place);

} // namespace dnd

#endif // _DND__NDARRAY_NODE_HPP_
