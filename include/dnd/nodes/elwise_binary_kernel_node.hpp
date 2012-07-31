//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__ELWISE_BINARY_KERNEL_NODE_HPP_
#define _DND__ELWISE_BINARY_KERNEL_NODE_HPP_

#include <dnd/nodes/ndarray_node.hpp>
#include <dnd/kernels/kernel_instance.hpp>
#include <dnd/shape_tools.hpp>

namespace dnd {

class elwise_binary_kernel_node : public ndarray_node {
    /* The number of dimensions in the result array */
    int m_ndim;
    /* The shape of the result array */
    dimvector m_shape;
    /** The data type */
    dtype m_dtype;
    /** Pointers to the operand nodes */
    ndarray_node_ptr m_opnodes[2];
    /** The computational kernel */
    kernel_instance<binary_operation_t> m_kernel;

    // Non-copyable
    elwise_binary_kernel_node(const elwise_binary_kernel_node&);
    elwise_binary_kernel_node& operator=(const elwise_binary_kernel_node&);

    elwise_binary_kernel_node(const dtype& dt, const ndarray_node_ptr& opnode0, const ndarray_node_ptr& opnode1)
        : m_dtype(dt), m_kernel()
    {
        m_opnodes[0] = opnode0;
        m_opnodes[1] = opnode1;

        broadcast_input_shapes(2, m_opnodes, &m_ndim, &m_shape);
    }

    elwise_binary_kernel_node(const dtype& dt, const ndarray_node_ptr& opnode0, const ndarray_node_ptr& opnode1,
                        const kernel_instance<binary_operation_t>& kernel)
        : m_dtype(dt), m_kernel(kernel)
    {
        m_opnodes[0] = opnode0;
        m_opnodes[1] = opnode1;

        broadcast_input_shapes(2, m_opnodes, &m_ndim, &m_shape);
    }

public:

    virtual ~elwise_binary_kernel_node() {
    }

    ndarray_node_category get_category() const
    {
        return elwise_node_category;
    }

    const dtype& get_dtype() const {
        return m_dtype;
    }

    int get_ndim() const {
        return m_ndim;
    }

    const intptr_t *get_shape() const
    {
        return m_shape.get();
    }

    uint32_t get_access_flags() const
    {
        // Readable, and inherit the immutable access flag of the operands
        return read_access_flag |
            (m_opnodes[0]->get_access_flags() & m_opnodes[1]->get_access_flags() & immutable_access_flag);
    }

    int get_nop() const {
        return 2;
    }

    ndarray_node *get_opnode(int i) const {
        return m_opnodes[i].get_node();
    }

    ndarray_node_ptr as_dtype(const dtype& dt,
                        dnd::assign_error_mode errmode, bool allow_in_place);

    ndarray_node_ptr apply_linear_index(
                    int ndim, const bool *remove_axis,
                    const intptr_t *start_index, const intptr_t *index_strides,
                    const intptr_t *shape,
                    bool allow_in_place);

    void get_binary_operation(intptr_t dst_fixedstride, intptr_t src0_fixedstride,
                                intptr_t src1_fixedstride,
                                kernel_instance<binary_operation_t>& out_kernel) const;

    const char *node_name() const {
        return "elementwise_binary_kernel";
    }

    friend ndarray_node_ptr make_elwise_binary_kernel_node_copy_kernel(const dtype& dt,
                        const ndarray_node_ptr& opnode0, const ndarray_node_ptr& opnode1,
                        const kernel_instance<binary_operation_t>& kernel);

    friend ndarray_node_ptr make_elwise_binary_kernel_node_steal_kernel(const dtype& dt,
                        const ndarray_node_ptr& opnode0, const ndarray_node_ptr& opnode1,
                        kernel_instance<binary_operation_t>& kernel);
};

ndarray_node_ptr make_elwise_binary_kernel_node_copy_kernel(const dtype& dt,
                    const ndarray_node_ptr& opnode0, const ndarray_node_ptr& opnode1,
                    const kernel_instance<binary_operation_t>& kernel);

ndarray_node_ptr make_elwise_binary_kernel_node_steal_kernel(const dtype& dt,
                    const ndarray_node_ptr& opnode0, const ndarray_node_ptr& opnode1,
                    kernel_instance<binary_operation_t>& kernel);

} // namespace dnd

#endif // _DND__ELWISE_BINARY_KERNEL_NODE_HPP_
