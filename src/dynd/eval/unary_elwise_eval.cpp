//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#if 0 // TODO reenable?

#include <deque>

#include <dynd/shape_tools.hpp>
#include <dynd/raw_iteration.hpp>
#include <dynd/eval/eval_engine.hpp>
#include <dynd/eval/unary_elwise_eval.hpp>
#include <dynd/kernels/buffered_unary_kernels.hpp>
#include <dynd/memblock/pod_memory_block.hpp>

using namespace std;
using namespace dynd;


template<class KernelType>
static ndarray_node_ptr initialize_dst_memblock(bool copy, const ndt::type& dst_tp, int ndim, const intptr_t *shape,
                    const int *axis_perm, uint32_t access_flags,
                    KernelType& operation,
                    const memory_block_ptr& src_data_memblock,
                    memory_block_ptr& out_dst_memblock, char *&out_originptr)
{
    ndarray_node_ptr result;

    if (dst_dt.get_memory_management() != blockref_memory_management) {
        result = make_strided_ndarray_node(dst_tp, ndim, shape, axis_perm,
                            access_flags, NULL, NULL);
        // Because we just allocated this buffer, we can write to it even though it
        // might be marked as readonly because the src memory block is readonly
        out_originptr = const_cast<char *>(result->get_readonly_originptr());
    } else {
        auxdata_kernel_api *api = operation.auxdata.get_kernel_api();
        if (!copy && api->supports_referencing_src_memory_blocks(operation.auxdata)) {
            // If the kernel can reference existing memory, add a blockref to the src data
            result = make_strided_ndarray_node(dst_tp, ndim, shape, axis_perm,
                            access_flags, &src_data_memblock, &src_data_memblock + 1);
            // Because we just allocated this buffer, we can write to it even though it
            // might be marked as readonly because the src memory block is readonly
            out_originptr = const_cast<char *>(result->get_readonly_originptr());
        } else {
            // Otherwise allocate a new memory block for the destination
            out_dst_memblock = make_pod_memory_block();
            api->set_dst_memory_block(operation.auxdata, out_dst_memblock.get());
            result = make_strided_ndarray_node(dst_tp, ndim, shape, axis_perm,
                            access_flags, &out_dst_memblock, &out_dst_memblock + 1);
            // Because we just allocated this buffer, we can write to it even though it
            // might be marked as readonly because the src memory block is readonly
            out_originptr = const_cast<char *>(result->get_readonly_originptr());
        }
    }

    return DYND_MOVE(result);
}

ndarray_node *dynd::eval::push_front_node_unary_kernels(ndarray_node* node,
                    const eval::eval_context *ectx,
                    std::deque<kernel_instance<unary_operation_pair_t>>& out_kernels,
                    std::deque<intptr_t>& out_element_sizes)
{
    const ndt::type& dt = node->get_type();

    switch (node->get_category()) {
        case strided_array_node_category:
            // The expression type kernels
            if (dt.get_kind() == expr_kind) {
                push_front_dtype_storage_to_value_kernels(dt, ectx, out_kernels, out_element_sizes);
            }
            return node;
        case elwise_node_category:
            if (node->get_nop() == 1) {
                // The expression type kernels
                if (dt.get_kind() == expr_kind) {
                    push_front_dtype_storage_to_value_kernels(dt, ectx, out_kernels, out_element_sizes);
                } else if (out_kernels.empty()) {
                    out_element_sizes.push_front(node->get_opnode(0)->get_type().value_type().get_data_size());
                }
                // The node's kernel
                out_element_sizes.push_front(dt.storage_type().get_data_size());
                out_kernels.push_front(kernel_instance<unary_operation_pair_t>());
                node->get_unary_specialization_operation(out_kernels.front());
                // The kernels from the operand
                return push_front_node_unary_kernels(node->get_opnode(0), ectx, out_kernels, out_element_sizes);
            } else {
                stringstream ss;
                ss << "evaluating this expression graph (which is further connected to a unary node) is not yet supported:\n";
                node->debug_print(ss);
                throw runtime_error(ss.str());
            }
            break;
        default: {
            stringstream ss;
            ss << "evaluating this expression graph (which is further connected to a unary node) is not yet supported:\n";
            node->debug_print(ss);
            throw runtime_error(ss.str());
        }
    }

}

ndarray_node_ptr dynd::eval::evaluate_strided_with_unary_kernel(ndarray_node *node, const eval::eval_context *DYND_UNUSED(ectx),
                                bool copy, uint32_t access_flags,
                                const ndt::type& dst_tp, kernel_instance<unary_operation_pair_t>& operation)
{
    const ndt::type& src_tp = node->get_type();
    ndarray_node_ptr result;
    int ndim = node->get_ndim();

    // Adjust the access flags, and force a copy if the access flags require it
    eval::process_access_flags(access_flags, node->get_access_flags(), copy);

    // For blockref result dtypes, this is the memblock
    // where the variable sized data goes
    memory_block_ptr dst_memblock;

    // Generate the axis_perm from the input strides, and use it to allocate the output
    shortvector<int> axis_perm(ndim);
    const intptr_t *node_strides = node->get_strides();
    char *result_originptr;
    strides_to_axis_perm(ndim, node_strides, axis_perm.get());

    result = initialize_dst_memblock(copy, dst_tp, ndim, node->get_shape(), axis_perm.get(),
                        access_flags, operation, node->get_data_memory_block(), dst_memblock, result_originptr);

    // Execute the kernel for all the elements
    raw_ndarray_iter<1,1> iter(node->get_ndim(), node->get_shape(),
                    result_originptr, result->get_strides(),
                    node->get_readonly_originptr(), node->get_strides());
    
    intptr_t innersize = iter.innersize();
    intptr_t dst_stride = iter.innerstride<0>();
    intptr_t src0_stride = iter.innerstride<1>();
    unary_specialization_t uspec = get_unary_specialization(dst_stride, dst_tp.get_data_size(),
                                                                src0_stride, src_tp.get_data_size());
    unary_operation_t kfunc = operation.specializations[uspec];
    if (innersize > 0) {
        do {
            kfunc(iter.data<0>(), dst_stride,
                        iter.data<1>(), src0_stride,
                        innersize, operation.auxdata);
        } while (iter.iternext());
    }

    // Finalize the destination memory block if it was a blockref dtype
    if (dst_memblock.get() != NULL) {
        memory_block_pod_allocator_api *api = get_memory_block_pod_allocator_api(dst_memblock.get());
        api->finalize(dst_memblock.get());
    }

    return DYND_MOVE(result);
}

ndarray_node_ptr dynd::eval::evaluate_unary_elwise_array(ndarray_node* node, const eval::eval_context *ectx, bool copy, uint32_t access_flags)
{
    // Chain the kernels together
    deque<kernel_instance<unary_operation_pair_t>> kernels;
    deque<intptr_t> element_sizes;

    ndarray_node *strided_node = push_front_node_unary_kernels(node, ectx, kernels, element_sizes);

    kernel_instance<unary_operation_pair_t> operation;
    make_buffered_chain_unary_kernel(kernels, element_sizes, operation);

    return evaluate_strided_with_unary_kernel(strided_node, ectx, copy, access_flags, node->get_type().value_type(), operation);
}

#endif // TODO reenable?

