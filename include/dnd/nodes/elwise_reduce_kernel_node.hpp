//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__ELWISE_REDUCE_KERNEL_NODE_HPP_
#define _DND__ELWISE_REDUCE_KERNEL_NODE_HPP_

#include <dnd/nodes/ndarray_node.hpp>
#include <dnd/kernels/kernel_instance.hpp>

namespace dnd {

class elwise_reduce_kernel_node : public ndarray_node {
    /** The dtype of the result this operation produces */
    dtype m_dtype;
    /** Pointer to the operand node */
    ndarray_node_ptr m_opnode;
    /** The computational kernel */
    kernel_instance<unary_operation_t> m_kernel;
    /** The number of dimensions in the result */
    int m_ndim;
    /** If true, this reduction is evaluated as right-associative instead of left */
    bool m_rightassoc;
    /** Whether all the dimensions of the operand are retained */
    bool m_keepdims;
    /** The shape of the result */
    dimvector m_shape;
    /** The axes along which to do reduction. NOTE: Its size is m_opnode->get_ndim(), not m_ndim */
    shortvector<dnd_bool> m_reduce_axes;

    // Non-copyable
    elwise_reduce_kernel_node(const elwise_reduce_kernel_node&);
    elwise_reduce_kernel_node& operator=(const elwise_reduce_kernel_node&);

    elwise_reduce_kernel_node(const dtype& dt, const ndarray_node_ptr& opnode, dnd_bool *reduce_axes, bool rightassoc, bool keepdims);

public:

    virtual ~elwise_reduce_kernel_node() {
    }

    ndarray_node_category get_category() const
    {
        return elwise_reduce_node_category;
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
        return "elwise_reduce_kernel";
    }

    friend ndarray_node_ptr make_elwise_reduce_kernel_node_copy_kernel(const dtype& dt, const ndarray_node_ptr& opnode,
                                                dnd_bool *reduce_axes, bool rightassoc, bool keepdims,
                                                const kernel_instance<unary_operation_t>& kernel);

    friend ndarray_node_ptr make_elwise_reduce_kernel_node_steal_kernel(const dtype& dt, const ndarray_node_ptr& opnode,
                                                dnd_bool *reduce_axes, bool rightassoc, bool keepdims,
                                                kernel_instance<unary_operation_t>& kernel);
};

ndarray_node_ptr make_elwise_reduce_kernel_node_copy_kernel(const dtype& dt, const ndarray_node_ptr& opnode,
                                            dnd_bool *reduce_axes, bool rightassoc, bool keepdims,
                                            const kernel_instance<unary_operation_t>& kernel);

ndarray_node_ptr make_elwise_reduce_kernel_node_steal_kernel(const dtype& dt, const ndarray_node_ptr& opnode,
                                            dnd_bool *reduce_axes, bool rightassoc, bool keepdims,
                                            kernel_instance<unary_operation_t>& kernel);

} // namespace dnd

#endif // _DND__ELWISE_REDUCE_KERNEL_NODE_HPP_
