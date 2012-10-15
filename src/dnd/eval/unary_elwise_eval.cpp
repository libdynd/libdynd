//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <deque>

#include <dnd/shape_tools.hpp>
#include <dnd/raw_iteration.hpp>
#include <dnd/eval/eval_engine.hpp>
#include <dnd/eval/unary_elwise_eval.hpp>
#include <dnd/kernels/buffered_unary_kernels.hpp>
#include <dnd/memblock/pod_memory_block.hpp>

using namespace std;
using namespace dynd;


template<class KernelType>
static ndarray_node_ptr initialize_dst_memblock(bool copy, const dtype& dst_dt, int ndim, const intptr_t *shape,
                    const int *axis_perm, uint32_t access_flags,
                    KernelType& operation,
                    const memory_block_ptr& src_data_memblock,
                    memory_block_ptr& out_dst_memblock, char *&out_originptr)
{
    ndarray_node_ptr result;

    if (dst_dt.get_memory_management() != blockref_memory_management) {
        result = make_strided_ndarray_node(dst_dt, ndim, shape, axis_perm,
                            access_flags, NULL, NULL);
        // Because we just allocated this buffer, we can write to it even though it
        // might be marked as readonly because the src memory block is readonly
        out_originptr = const_cast<char *>(result->get_readonly_originptr());
    } else {
        auxdata_kernel_api *api = operation.auxdata.get_kernel_api();
        if (!copy && api->supports_referencing_src_memory_blocks(operation.auxdata)) {
            // If the kernel can reference existing memory, add a blockref to the src data
            result = make_strided_ndarray_node(dst_dt, ndim, shape, axis_perm,
                            access_flags, &src_data_memblock, &src_data_memblock + 1);
            // Because we just allocated this buffer, we can write to it even though it
            // might be marked as readonly because the src memory block is readonly
            out_originptr = const_cast<char *>(result->get_readonly_originptr());
        } else {
            // Otherwise allocate a new memory block for the destination
            out_dst_memblock = make_pod_memory_block();
            api->set_dst_memory_block(operation.auxdata, out_dst_memblock.get());
            result = make_strided_ndarray_node(dst_dt, ndim, shape, axis_perm,
                            access_flags, &out_dst_memblock, &out_dst_memblock + 1);
            // Because we just allocated this buffer, we can write to it even though it
            // might be marked as readonly because the src memory block is readonly
            out_originptr = const_cast<char *>(result->get_readonly_originptr());
        }
    }

    return DND_MOVE(result);
}

ndarray_node *dynd::eval::push_front_node_unary_kernels(ndarray_node* node,
                    const eval::eval_context *ectx,
                    std::deque<unary_specialization_kernel_instance>& out_kernels,
                    std::deque<intptr_t>& out_element_sizes)
{
    const dtype& dt = node->get_dtype();

    switch (node->get_category()) {
        case strided_array_node_category:
            // The dtype expression kernels
            if (dt.kind() == expression_kind) {
                push_front_dtype_storage_to_value_kernels(dt, ectx, out_kernels, out_element_sizes);
            }
            return node;
        case elwise_node_category:
            if (node->get_nop() == 1) {
                // The dtype expression kernels
                if (dt.kind() == expression_kind) {
                    push_front_dtype_storage_to_value_kernels(dt, ectx, out_kernels, out_element_sizes);
                } else if (out_kernels.empty()) {
                    out_element_sizes.push_front(node->get_opnode(0)->get_dtype().value_dtype().element_size());
                }
                // The node's kernel
                out_element_sizes.push_front(dt.storage_dtype().element_size());
                out_kernels.push_front(unary_specialization_kernel_instance());
                node->get_unary_specialization_operation(out_kernels.front());
                // The kernels from the operand
                return push_front_node_unary_kernels(node->get_opnode(0), ectx, out_kernels, out_element_sizes);
            } else {
                stringstream ss;
                ss << "evaluating this expression graph (which is further connected to a unary node) is not yet supported:\n";
                node->debug_dump(ss);
                throw runtime_error(ss.str());
            }
            break;
        default: {
            stringstream ss;
            ss << "evaluating this expression graph (which is further connected to a unary node) is not yet supported:\n";
            node->debug_dump(ss);
            throw runtime_error(ss.str());
        }
    }

}

ndarray_node_ptr dynd::eval::evaluate_strided_with_unary_kernel(ndarray_node *node, const eval::eval_context *DND_UNUSED(ectx),
                                bool copy, uint32_t access_flags,
                                const dtype& dst_dt, unary_specialization_kernel_instance& operation)
{
    const dtype& src_dt = node->get_dtype();
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

    result = initialize_dst_memblock(copy, dst_dt, ndim, node->get_shape(), axis_perm.get(),
                        access_flags, operation, node->get_data_memory_block(), dst_memblock, result_originptr);

    // Execute the kernel for all the elements
    raw_ndarray_iter<1,1> iter(node->get_ndim(), node->get_shape(),
                    result_originptr, result->get_strides(),
                    node->get_readonly_originptr(), node->get_strides());
    
    intptr_t innersize = iter.innersize();
    intptr_t dst_stride = iter.innerstride<0>();
    intptr_t src0_stride = iter.innerstride<1>();
    unary_specialization_t uspec = get_unary_specialization(dst_stride, dst_dt.element_size(),
                                                                src0_stride, src_dt.element_size());
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

    return DND_MOVE(result);
}

ndarray_node_ptr dynd::eval::evaluate_unary_elwise_array(ndarray_node* node, const eval::eval_context *ectx, bool copy, uint32_t access_flags)
{
    // Chain the kernels together
    deque<unary_specialization_kernel_instance> kernels;
    deque<intptr_t> element_sizes;

    ndarray_node *strided_node = push_front_node_unary_kernels(node, ectx, kernels, element_sizes);

    unary_specialization_kernel_instance operation;
    make_buffered_chain_unary_kernel(kernels, element_sizes, operation);

    return evaluate_strided_with_unary_kernel(strided_node, ectx, copy, access_flags, node->get_dtype().value_dtype(), operation);
}

