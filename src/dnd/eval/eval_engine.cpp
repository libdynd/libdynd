//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <deque>

#include <dnd/eval/eval_engine.hpp>
#include <dnd/shape_tools.hpp>
#include <dnd/raw_iteration.hpp>
#include <dnd/kernels/assignment_kernels.hpp>
#include <dnd/kernels/buffered_unary_kernels.hpp>
#include <dnd/nodes/elwise_reduce_kernel_node.hpp>
#include <dnd/memblock/pod_memory_block.hpp>

using namespace std;
using namespace dnd;


static ndarray_node_ptr copy_strided_array(ndarray_node* node, uint32_t access_flags)
{
    int ndim = node->get_ndim();
    const dtype& dt = node->get_dtype();
    dtype_memory_management_t mem_mgmt = dt.get_memory_management();

    // Default to readwrite if no flags were specified
    if (access_flags == 0) {
        access_flags = read_access_flag|write_access_flag;
    }

    // Sort the strides to get the memory layout ordering
    shortvector<int> axis_perm(ndim);
    strides_to_axis_perm(ndim, node->get_strides(), axis_perm.get());

    // Create the blockrefs for variable sized data if needed
    memory_block_ptr *blockrefs_begin = NULL, *blockrefs_end = NULL;
    memory_block_ptr dst_memblock;
    if (mem_mgmt == blockref_memory_management) {
        // TODO: This will need to replicate the full nested blockref structure
        //       of the dtype, probably the logic to do that goes into its own
        //       function somewhere.
        dst_memblock = make_pod_memory_block();
        blockrefs_begin = &dst_memblock;
        blockrefs_end = &dst_memblock + 1;
    }

    // Construct the new array
    ndarray_node_ptr result = make_strided_ndarray_node(dt, ndim,
                                    node->get_shape(), axis_perm.get(), access_flags, blockrefs_begin, blockrefs_end);

    // Get the kernel copy operation
    unary_specialization_kernel_instance kernel;
    get_dtype_assignment_kernel(dt, kernel);
    if (mem_mgmt == blockref_memory_management) {
        // Set up the destination memory block for the blockref copy kernel
        auxdata_kernel_api *api = static_cast<AuxDataBase *>(kernel.auxdata)->kernel_api;
        if (api == NULL) {
            stringstream ss;
            ss << "internal error: assignment kernel for dtype " << dt << " did not provide a kernel API in the auxdata";
            throw runtime_error(ss.str());
        }
        api->set_dst_memory_block(kernel.auxdata, dst_memblock.get());
    }

    // Do the actual copy operation. We get the readonly pointer and const_cast it,
    // because the permissions may not allow writeability, but we created the node
    // for the first time just now so we want to write to it once.
    raw_ndarray_iter<1,1> iter(result->get_ndim(), result->get_shape(),
                                    const_cast<char *>(result->get_readonly_originptr()), result->get_strides(),
                            node->get_readonly_originptr(), node->get_strides());
    intptr_t innersize = iter.innersize();
    intptr_t dst_innerstride = iter.innerstride<0>(), src_innerstride = iter.innerstride<1>();
    unary_operation_t assign_fn = kernel.specializations[
        get_unary_specialization(dst_innerstride, dt.element_size(), src_innerstride, dt.element_size())];
    if (innersize > 0) {
        do {
            assign_fn(iter.data<0>(), dst_innerstride,
                        iter.data<1>(), src_innerstride,
                        innersize, kernel.auxdata);
        } while (iter.iternext());
    }

    // Finalize the destination memory block if it was a blockref dtype
    if (dst_memblock.get() != NULL) {
        memory_block_pod_allocator_api *api = get_memory_block_pod_allocator_api(dst_memblock.get());
        api->finalize(dst_memblock.get());
    }

    return result;
}

/**
 * Analyzes whether a copy is required from the src to the dst because of the permissions.
 * Sets the dst_access_flags, and flips out_copy_required to true when a copy is needed.
 */
static void process_access_flags(uint32_t &dst_access_flags, uint32_t src_access_flags, bool &inout_copy_required)
{
    if (dst_access_flags != 0 && dst_access_flags != src_access_flags) {
        if (dst_access_flags&write_access_flag) {
            // If writeable is requested, and src isn't writeable, must copy
            if (!(src_access_flags&write_access_flag)) {
                inout_copy_required = true;
            }
        } else if (src_access_flags&immutable_access_flag) {
            // Always propagate the immutable flag from src to dst
            if (!inout_copy_required) {
                dst_access_flags |= immutable_access_flag;
            }
        } else if (dst_access_flags&immutable_access_flag) {
            // If immutable is requested, and src isn't immutable, must copy
            if (!(src_access_flags&immutable_access_flag)) {
                inout_copy_required = true;
            }
        }
    }
}

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

static ndarray_node_ptr evaluate_strided_array_kernel(ndarray_node *node, const eval_context *DND_UNUSED(ectx),
                                bool copy, uint32_t access_flags,
                                const dtype& dst_dt, unary_specialization_kernel_instance& operation)
{
    const dtype& src_dt = node->get_dtype();
    ndarray_node_ptr result;
    int ndim = node->get_ndim();

    // Adjust the access flags, and force a copy if the access flags require it
    process_access_flags(access_flags, node->get_access_flags(), copy);

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

static ndarray_node_ptr evaluate_strided_array_expression_dtype(ndarray_node* node, const eval_context *ectx, bool copy, uint32_t access_flags)
{
    const dtype& dt = node->get_dtype();
    const dtype& value_dt = dt.value_dtype();
    ndarray_node_ptr result;
    //int ndim = node->get_ndim();

    unary_specialization_kernel_instance operation;
    get_dtype_assignment_kernel(value_dt, dt, assign_error_none, ectx, operation);

    return evaluate_strided_array_kernel(node, ectx, copy, access_flags, value_dt, operation);
}

static ndarray_node *push_front_unary_kernels(ndarray_node* node,
                    const eval_context *ectx,
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
                return push_front_unary_kernels(node->get_opnode(0), ectx, out_kernels, out_element_sizes);
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

static ndarray_node_ptr evaluate_unary_elwise_array(ndarray_node* node, const eval_context *ectx, bool copy, uint32_t access_flags)
{
    ndarray_node *op = node->get_opnode(0);
    const dtype& dt = op->get_dtype();

    // Chain the kernels together
    deque<unary_specialization_kernel_instance> kernels;
    deque<intptr_t> element_sizes;

    ndarray_node *strided_node = push_front_unary_kernels(node, ectx, kernels, element_sizes);

    unary_specialization_kernel_instance operation;
    make_buffered_chain_unary_kernel(kernels, element_sizes, operation);

    return evaluate_strided_array_kernel(strided_node, ectx, copy, access_flags, dt.value_dtype(), operation);
}

static ndarray_node_ptr evaluate_binary_elwise_array(ndarray_node* node, const eval_context *ectx, bool DND_UNUSED(copy), uint32_t access_flags)
{
    ndarray_node *op1 = node->get_opnode(0);
    ndarray_node *op2 = node->get_opnode(1);

    // Special case of two strided sub-operands, requiring no intermediate buffers
    if (op1->get_category() == strided_array_node_category &&
                op2->get_category() == strided_array_node_category) {
        ndarray_node_ptr result;
        raw_ndarray_iter<1,2> iter(node->get_ndim(), node->get_shape(),
                                    node->get_dtype().value_dtype(), result, access_flags,
                                    op1, op2);
        //iter.debug_dump(std::cout);

        intptr_t innersize = iter.innersize();
        intptr_t dst_stride = iter.innerstride<0>();
        intptr_t src0_stride = iter.innerstride<1>();
        intptr_t src1_stride = iter.innerstride<2>();
        kernel_instance<binary_operation_t> operation;
        node->get_binary_operation(dst_stride, src0_stride, src1_stride, ectx, operation);
        if (innersize > 0) {
            do {
                operation.kernel(iter.data<0>(), dst_stride,
                            iter.data<1>(), src0_stride,
                            iter.data<2>(), src1_stride,
                            innersize, operation.auxdata);
            } while (iter.iternext());
        }

        return DND_MOVE(result);
    }

    stringstream ss;
    ss << "evaluating this expression graph is not yet supported:\n";
    node->debug_dump(ss);
    throw runtime_error(ss.str());
}

/**
 * Creates a result array for an elementwise
 * reduce operation.
 */
static ndarray_node_ptr make_elwise_reduce_result(const dtype& result_dt, uint32_t access_flags, bool keepdims,
                            int ndim, const dnd_bool *reduce_axes, const intptr_t *src_shape, const int *src_axis_perm,
                            char *&result_originptr, intptr_t *result_strides)
{
    dimvector result_shape(ndim);

    // Calculate the shape and strides of the reduction result
    // without removing the dimensions
    intptr_t num_elements = 1;
    intptr_t stride = result_dt.element_size();
    for (int i = 0; i < ndim; ++i) {
        int p = src_axis_perm[i];
        if (reduce_axes[p]) {
            result_shape[p] = 1;
            result_strides[p] = 0;
        } else {
            intptr_t size = src_shape[p];
            result_shape[p] = size;
            if (size == 1) {
                result_strides[p] = 0;
            } else {
                result_strides[p] = stride;
                stride *= size;
                num_elements *= size;
            }
        }
    }

    // Allocate the memoryblock for the data
    char *originptr = NULL;
    memory_block_ptr memblock = make_fixed_size_pod_memory_block(result_dt.element_size() * num_elements,
                    result_dt.alignment(), &originptr,
                    NULL, NULL);

    ndarray_node_ptr result;

    // Create the strided ndarray node, compressing the dimensions if requested
    if (!keepdims) {
        dimvector compressed_shape(ndim), compressed_strides(ndim);
        int compressed_ndim = 0;
        for (int i = 0; i < ndim; ++i) {
            if (!reduce_axes[i]) {
                compressed_shape[compressed_ndim] = result_shape[i];
                compressed_strides[compressed_ndim] = result_strides[i];
                ++compressed_ndim;
            }
        }
        result = make_strided_ndarray_node(result_dt, compressed_ndim,
                    compressed_shape.get(), compressed_strides.get(), originptr, access_flags, memblock);
    } else {
        result = make_strided_ndarray_node(result_dt, ndim,
                    result_shape.get(), result_strides, originptr, access_flags, memblock);
    }
    // Because we just allocated this buffer, we can write to it even though it
    // might be marked as readonly because the src memory block is readonly
    result_originptr = const_cast<char *>(result->get_readonly_originptr());

    return DND_MOVE(result);
}

static ndarray_node_ptr evaluate_elwise_reduce_array(ndarray_node* node, const eval_context *ectx, bool copy, uint32_t access_flags)
{
    elwise_reduce_kernel_node *rnode = static_cast<elwise_reduce_kernel_node*>(node);
    ndarray_node *strided_node = rnode->get_opnode(0);

    const dtype& result_dt = rnode->get_dtype().value_dtype();

    if (result_dt.get_memory_management() == blockref_memory_management) {
        throw runtime_error("blockref memory management isn't supported for elwise reduce gfuncs yet");
    }

    // Used when the input
    deque<unary_specialization_kernel_instance> kernels;
    deque<intptr_t> element_sizes;

    if (strided_node->get_category() != strided_array_node_category ||
                    strided_node->get_dtype().kind() == expression_kind) {
        strided_node = push_front_unary_kernels(strided_node, ectx, kernels, element_sizes);
    }

    // Adjust the access flags, and force a copy if the access flags require it
    process_access_flags(access_flags, node->get_access_flags(), copy);

    int src_ndim = strided_node->get_ndim();
    const char *src_originptr = strided_node->get_readonly_originptr();
    const intptr_t *src_shape = strided_node->get_shape();
    dimvector adjusted_src_strides(src_ndim);
    memcpy(adjusted_src_strides.get(), strided_node->get_strides(), sizeof(intptr_t) * src_ndim);

    // Generate the axis_perm from the input strides, and use it to allocate the output
    shortvector<int> axis_perm(src_ndim);
    strides_to_axis_perm(src_ndim, adjusted_src_strides.get(), axis_perm.get());

    char *result_originptr;
    dimvector result_strides(src_ndim);
    const dnd_bool *reduce_axes = rnode->get_reduce_axes();

    ndarray_node_ptr result = make_elwise_reduce_result(result_dt, access_flags,
                            rnode->get_keepdims(),
                            src_ndim, reduce_axes, src_shape, axis_perm.get(),
                            result_originptr, result_strides.get());

    // If we're doing a right associative reduce, reverse the reduction axes
    if (rnode->get_rightassoc()) {
        for (int i = 0; i < src_ndim; ++i) {
            if (reduce_axes[i]) {
                src_originptr += (src_shape[i] - 1) * adjusted_src_strides[i];
                adjusted_src_strides[i] = -adjusted_src_strides[i];
            }
        }
    }

    // Initialize the reduction result
    dimvector adjusted_src_shape(src_ndim);

    if (rnode->get_identity()) {
        // Copy the identity scalar to the whole result
        intptr_t result_count = 1;
        const intptr_t *result_shape = result->get_shape();
        for (int i = 0, i_end = result->get_ndim(); i != i_end; ++i) {
            result_count *= result_shape[i];
        }
        unary_specialization_kernel_instance copy_kernel;
        get_dtype_assignment_kernel(result_dt, copy_kernel);
        unary_operation_t copy_op = copy_kernel.specializations[scalar_to_contiguous_unary_specialization];
        copy_op(const_cast<char *>(result->get_readonly_originptr()), result_dt.element_size(),
                        rnode->get_identity()->get_readonly_originptr(), 0,
                        result_count, copy_kernel.auxdata);

        // No change to the src shape
        memcpy(adjusted_src_shape.get(), src_shape, sizeof(intptr_t) * src_ndim);
    } else {
        // Copy the first element along each reduction dimension, then exclude it
        // from the later reduction loop
        unary_specialization_kernel_instance copy_kernel;
        if (kernels.empty()) {
            // Straightforward copy kernel
            get_dtype_assignment_kernel(result_dt, copy_kernel);
        } else {
            // Borrow all the kernels so we can make a copy kernel for this part
            deque<unary_specialization_kernel_instance> borrowed_kernels(kernels.size());
            for (size_t i = 0, i_end = kernels.size(); i != i_end; ++i) {
                borrowed_kernels[i].borrow_from(kernels[i]);
            }
            make_buffered_chain_unary_kernel(borrowed_kernels, element_sizes, copy_kernel);
        }

        // Create the shape for the result and
        // the src shape with the first reduce elements cut out
        dimvector result_shape(src_ndim), tmp_src_strides(src_ndim);
        for (int i = 0; i < src_ndim; ++i) {
            if (reduce_axes[i]) {
                result_shape[i] = 1;
                adjusted_src_shape[i] = max(src_shape[i] - 1, (intptr_t)0);
                tmp_src_strides[i] = 0;
            } else {
                result_shape[i] = src_shape[i];
                adjusted_src_shape[i] = src_shape[i];
                tmp_src_strides[i] = adjusted_src_strides[i];
            }
        }

        // Set up the iterator for the copy
        raw_ndarray_iter<1,1> iter(src_ndim, result_shape.get(), result_originptr, result_strides.get(),
                                    src_originptr, tmp_src_strides.get(), axis_perm.get());
        intptr_t innersize = iter.innersize();
        intptr_t dst_stride = iter.innerstride<0>();
        intptr_t src0_stride = iter.innerstride<1>();
        unary_specialization_t uspec = get_unary_specialization(dst_stride, result_dt.element_size(),
                                                    src0_stride, strided_node->get_dtype().storage_dtype().element_size());
        unary_operation_t copy_op = copy_kernel.specializations[uspec];
        if (innersize > 0) {
            do {
                copy_op(iter.data<0>(), dst_stride,
                            iter.data<1>(), src0_stride,
                            innersize, copy_kernel.auxdata);
            } while (iter.iternext());
        }

        // Adjust the src origin pointer to skip the first reduce elements
        for (int i = 0; i < src_ndim; ++i) {
            if (reduce_axes[i]) {
                src_originptr += adjusted_src_strides[i];
            }
        }
    }

    // Set up the iterator
    raw_ndarray_iter<1,1> iter(src_ndim, adjusted_src_shape.get(),
                    result_originptr, result_strides.get(),
                    src_originptr, adjusted_src_strides.get());
    
    // Get the kernel to use in the inner loop
    kernel_instance<unary_operation_t> reduce_operation;
    unary_operation_t reduce_op_duped[4];

    intptr_t innersize = iter.innersize();
    intptr_t dst_stride = iter.innerstride<0>();
    intptr_t src0_stride = iter.innerstride<1>();
    unary_specialization_t uspec = get_unary_specialization(dst_stride, result_dt.element_size(),
                                                src0_stride, strided_node->get_dtype().storage_dtype().element_size());

    // Create the reduction kernel
    rnode->get_unary_operation(reduce_operation);
    if (!kernels.empty()) {
        // Create a unary specialization kernel by replicating the general kernel
        element_sizes.push_back(node->get_dtype().element_size());
        reduce_op_duped[0] = reduce_operation.kernel;
        reduce_op_duped[1] = reduce_operation.kernel;
        reduce_op_duped[2] = reduce_operation.kernel;
        reduce_op_duped[3] = reduce_operation.kernel;
        kernels.push_back(unary_specialization_kernel_instance());
        kernels.back().specializations = reduce_op_duped;
        kernels.back().auxdata.swap(reduce_operation.auxdata);

        // Create the chained kernel
        unary_specialization_kernel_instance chained_kernel;
        make_buffered_chain_unary_kernel(kernels, element_sizes, chained_kernel);
        // Pick out the right specialization
        reduce_operation.kernel = chained_kernel.specializations[uspec];
        reduce_operation.auxdata.swap(chained_kernel.auxdata);
    }

    if (innersize > 0) {
        do {
            reduce_operation.kernel(iter.data<0>(), dst_stride,
                        iter.data<1>(), src0_stride,
                        innersize, reduce_operation.auxdata);
        } while (iter.iternext());
    }

    return DND_MOVE(result);
}

ndarray_node_ptr dnd::evaluate(ndarray_node *node, const eval_context *ectx, bool copy, uint32_t access_flags)
{
    if ((access_flags&(immutable_access_flag|write_access_flag)) == (immutable_access_flag|write_access_flag)) {
        throw runtime_error("Cannot create an ndarray which is both writeable and immutable");
    }

    switch (node->get_category()) {
        case strided_array_node_category:
            if (node->get_dtype().kind() != expression_kind) {
                if (!copy && (access_flags == 0 || access_flags == node->get_access_flags() ||
                                (access_flags == read_access_flag &&
                                 node->get_access_flags() == (read_access_flag|immutable_access_flag)))) {
                    // If no copy is requested, can avoid a copy when the access flags
                    // match, or if just readonly is requested but src is also immutable.
                    return node->as_ndarray_node_ptr();
                } else {
                    return copy_strided_array(node, access_flags);
                }
            } else {
                return evaluate_strided_array_expression_dtype(node, ectx, copy, access_flags);
            }
            break;
        case elwise_node_category: {
            switch (node->get_nop()) {
                case 1:
                    return evaluate_unary_elwise_array(node, ectx, copy, access_flags);
                case 2:
                    return evaluate_binary_elwise_array(node, ectx, copy, access_flags);
                default:
                    break;
            }
        }
        case elwise_reduce_node_category:
            return evaluate_elwise_reduce_array(node, ectx, copy, access_flags);
        case arbitrary_node_category:
            throw std::runtime_error("evaluate is not yet implemented for"
                            " nodes with an arbitrary_node_category category");
    }

    stringstream ss;
    ss << "Evaluating the following node is not yet supported:\n";
    node->debug_dump(ss);
    throw std::runtime_error(ss.str());
}
