//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__NDARRAY_NODE_HPP_
#define _DND__NDARRAY_NODE_HPP_

#include <iostream>

#include <boost/detail/atomic_count.hpp>
#include <boost/intrusive_ptr.hpp>

#include <dnd/dtype.hpp>
#include <dnd/dtype_assign.hpp>
#include <dnd/irange.hpp>
#include <dnd/kernels/kernel_instance.hpp>
#include <dnd/shortvector.hpp>
#include <dnd/memblock/memory_block.hpp>

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

class ndarray_node_ptr;

/**
 * Virtual base class for the ndarray expression tree.
 */
class ndarray_node {
    // Non-copyable
    ndarray_node(const ndarray_node&);
    ndarray_node& operator=(const ndarray_node&);

protected:

    ndarray_node() {
    }

public:

    virtual ~ndarray_node() {
    }

    /**
     * The ndarray_node is always allocated within a memory_block
     * object. This returns the memory block smart pointer.
     */
    memory_block_ptr as_memory_block_ptr();

    /**
     * The ndarray_node is always allocated within a node memory_block
     * object. This returns the ndarray node smart pointer.
     */
    ndarray_node_ptr as_ndarray_node_ptr();

    /**
     * The data for a strided node is always stored in a memory block,
     * which may be this node or some other memory block. This
     * returns the memory block which stores the data.
     *
     * For nodes that are expression-based, this returns NULL.
     *
     * In the case of blockref dtypes, this memory block also
     * holds references to those other memory blocks.
     */
    virtual memory_block_ptr get_data_memory_block();

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

    /** The number of operand nodes this node depends on */
    virtual int get_nop() const {
        return 0;
    }

    virtual const ndarray_node_ptr& get_opnode(int DND_UNUSED(i)) const {
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
    ndarray_node_ptr evaluate();

    /**
     * Converts this node to a new dtype. This uses a convert_dtype.
     */
    virtual ndarray_node_ptr as_dtype(const dtype& dt,
                        assign_error_mode errmode = assign_error_default, bool allow_in_place = false) = 0;

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
    virtual ndarray_node_ptr apply_linear_index(
                    int ndim, const bool *remove_axis,
                    const intptr_t *start_index, const intptr_t *index_strides,
                    const intptr_t *shape,
                    bool allow_in_place) = 0;

    /** Debug printing of the tree */
    void debug_dump(std::ostream& o, const std::string& indent = "") const;
    /** Debug printing of the data from the derived class */
    virtual void debug_dump_extra(std::ostream& o, const std::string& indent) const;

    virtual const char *node_name() const = 0;

    friend void intrusive_ptr_add_ref(const ndarray_node *node);
    friend void intrusive_ptr_release(const ndarray_node *node);
};

/**
 * A reference-counted smart pointer for ndarray nodes,
 * which are themselves specific instances of memory_blocks.
 */
class ndarray_node_ptr : public memory_block_ptr {
public:
    /** Default constructor */
    ndarray_node_ptr()
        : memory_block_ptr()
    {
    }

    /** Constructor from a raw pointer */
    explicit ndarray_node_ptr(memory_block_data *memblock, bool add_ref = true)
        : memory_block_ptr(memblock, add_ref)
    {
        if (memblock->m_type != ndarray_node_memory_block_type) {
            throw std::runtime_error("Can only make an ndarray_node_ptr from an ndarray node memory_block");
        }
    }

    /** Copy constructor */
    ndarray_node_ptr(const ndarray_node_ptr& rhs)
        : memory_block_ptr(rhs)
    {
    }

#ifdef DND_RVALUE_REFS
    /** Move constructor */
    ndarray_node_ptr(ndarray_node_ptr&& rhs)
        : memory_block_ptr(DND_MOVE(rhs))
    {
    }
#endif

    /** Assignment */
    ndarray_node_ptr& operator=(const ndarray_node_ptr& rhs)
    {
        *static_cast<memory_block_ptr *>(this) = static_cast<const memory_block_ptr&>(rhs);
        return *this;
    }

    /** Move assignment */
#ifdef DND_RVALUE_REFS
    ndarray_node_ptr& operator=(ndarray_node_ptr&& rhs)
    {
        *static_cast<memory_block_ptr *>(this) = DND_MOVE(static_cast<memory_block_ptr&&>(rhs));
        return *this;
    }
#endif

    void swap(ndarray_node_ptr& rhs) {
        static_cast<memory_block_ptr *>(this)->swap(static_cast<memory_block_ptr&>(rhs));
    }

    /** This object behaves like an ndarray_node pointer */
    ndarray_node *operator->() const {
        return reinterpret_cast<ndarray_node *>(reinterpret_cast<char *>(get()) + sizeof(memory_block_data));
    }
};

/** Applies the slicing index to the ndarray node. */
ndarray_node_ptr apply_index_to_node(const ndarray_node_ptr& node,
                                int nindex, const irange *indices, bool allow_in_place);
/**
 * Applies an integer index to the ndarray node.
 */
ndarray_node_ptr apply_integer_index_to_node(const ndarray_node_ptr& node,
                                int axis, intptr_t idx, bool allow_in_place);

inline memory_block_ptr ndarray_node::as_memory_block_ptr()
{
    // Subtract to get the memory_block pointer
    return memory_block_ptr(reinterpret_cast<memory_block_data *>(
                reinterpret_cast<char *>(this) - sizeof(memory_block_data)));
}

inline ndarray_node_ptr ndarray_node::as_ndarray_node_ptr()
{
    // Subtract to get the memory_block pointer
    return ndarray_node_ptr(reinterpret_cast<memory_block_data *>(
                reinterpret_cast<char *>(this) - sizeof(memory_block_data)));
}

} // namespace dnd

#endif // _DND__NDARRAY_NODE_HPP_
