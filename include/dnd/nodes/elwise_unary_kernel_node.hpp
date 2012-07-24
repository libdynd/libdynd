//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__ELWISE_UNARY_KERNEL_NODE_HPP_
#define _DND__ELWISE_UNARY_KERNEL_NODE_HPP_

#include <dnd/nodes/ndarray_node.hpp>
#include <dnd/kernels/unary_kernel_instance.hpp>

namespace dnd {

class elwise_unary_kernel_node : public ndarray_node {
    /** The data type */
    dtype m_dtype;
    /** Pointer to the operand node */
    ndarray_node_ptr m_opnode;
    /** The computational kernel */
    unary_specialization_kernel_instance m_kernel;

    // Non-copyable
    elwise_unary_kernel_node(const elwise_unary_kernel_node&);
    elwise_unary_kernel_node& operator=(const elwise_unary_kernel_node&);

    elwise_unary_kernel_node(const dtype& dt, const ndarray_node_ptr& opnode)
        : m_dtype(dt), m_opnode(opnode), m_kernel()
    {
    }

    elwise_unary_kernel_node(const dtype& dt, const ndarray_node_ptr& opnode, const unary_specialization_kernel_instance& kernel)
        : m_dtype(dt), m_opnode(opnode), m_kernel(kernel)
    {
    }

public:

    virtual ~elwise_unary_kernel_node() {
    }

    ndarray_node_category get_category() const
    {
        return elementwise_node_category;
    }

    const dtype& get_dtype() const {
        return m_dtype;
    }

    int get_ndim() const {
        return m_opnode->get_ndim();
    }

    const intptr_t *get_shape() const
    {
        return m_opnode->get_shape();
    }

    uint32_t get_access_flags() const
    {
        // Readable, and inherit the immutable access flag of the operand
        return read_access_flag |
            (m_opnode->get_access_flags() & immutable_access_flag);
    }

    int get_nop() const {
        return 1;
    }

    const ndarray_node_ptr& get_opnode(int i) const {
        return m_opnode;
    }

    ndarray_node_ptr as_dtype(const dtype& dt,
                        dnd::assign_error_mode errmode, bool allow_in_place);

    ndarray_node_ptr apply_linear_index(
                    int ndim, const bool *remove_axis,
                    const intptr_t *start_index, const intptr_t *index_strides,
                    const intptr_t *shape,
                    bool allow_in_place);

    void get_unary_operation(intptr_t dst_fixedstride, intptr_t src_fixedstride,
                                kernel_instance<unary_operation_t>& out_kernel) const;

    const char *node_name() const {
        return "elementwise_unary_kernel";
    }

    friend ndarray_node_ptr make_elwise_unary_kernel_node_copy_kernel(const dtype& dt, const ndarray_node_ptr& opnode,
                                                const unary_specialization_kernel_instance& kernel);

    friend ndarray_node_ptr make_elwise_unary_kernel_node_steal_kernel(const dtype& dt, const ndarray_node_ptr& opnode,
                                                unary_specialization_kernel_instance& kernel);
};

ndarray_node_ptr make_elwise_unary_kernel_node_copy_kernel(const dtype& dt, const ndarray_node_ptr& opnode,
                                            const unary_specialization_kernel_instance& kernel);

ndarray_node_ptr make_elwise_unary_kernel_node_steal_kernel(const dtype& dt, const ndarray_node_ptr& opnode,
                                            unary_specialization_kernel_instance& kernel);

} // namespace dnd

#endif // _DND__ELWISE_UNARY_KERNEL_NODE_HPP_
