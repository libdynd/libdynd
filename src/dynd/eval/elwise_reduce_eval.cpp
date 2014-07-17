//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#if 0 // TODO reenable?

#include <dynd/raw_iteration.hpp>
#include <dynd/eval/eval_engine.hpp>
#include <dynd/eval/unary_elwise_eval.hpp>
#include <dynd/eval/elwise_reduce_eval.hpp>
#include <dynd/eval/groupby_elwise_reduce_eval.hpp>
#include <dynd/kernels/buffered_unary_kernels.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/memblock/fixed_size_pod_memory_block.hpp>

using namespace std;
using namespace dynd;

/**
 * Creates a result array for an elementwise
 * reduce operation.
 */
static ndarray_node_ptr make_elwise_reduce_result(const ndt::type& result_dt, uint32_t access_flags, bool keepdims,
                            int ndim, const dynd_bool *reduce_axes, const intptr_t *src_shape, const int *src_axis_perm,
                            char *&result_originptr, intptr_t *result_strides)
{
    dimvector result_shape(ndim);

    // Calculate the shape and strides of the reduction result
    // without removing the dimensions
    intptr_t num_elements = 1;
    intptr_t stride = result_dt.get_data_size();
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
    memory_block_ptr memblock = make_fixed_size_pod_memory_block(result_dt.get_data_size() * num_elements,
                    result_dt.get_data_alignment(), &originptr,
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

    return DYND_MOVE(result);
}


ndarray_node_ptr dynd::eval::evaluate_elwise_reduce_array(ndarray_node* node,
                    const eval::eval_context *ectx, bool copy, uint32_t access_flags)
{
    elwise_reduce_kernel_node *rnode = static_cast<elwise_reduce_kernel_node*>(node);
    ndarray_node *strided_node = rnode->get_opnode(0);

    const ndt::type& result_dt = rnode->get_type().value_type();

    if (result_dt.get_memory_management() == blockref_memory_management) {
        throw runtime_error("blockref memory management isn't supported for elwise reduce gfuncs yet");
    }

    // Used when the input is some kind of expression
    deque<kernel_instance<unary_operation_pair_t>> kernels;
    deque<intptr_t> element_sizes;

    if (strided_node->get_category() != strided_array_node_category ||
                    strided_node->get_type().get_kind() == expr_kind) {
        // If the next node is a groupby, call the special groupby reduction code
        if (strided_node->get_category() == groupby_node_category) {
            return evaluate_groupby_elwise_reduce(node, ectx, copy, access_flags);
        }
        strided_node = push_front_node_unary_kernels(strided_node, ectx, kernels, element_sizes);
    }

    // Adjust the access flags, and force a copy if the access flags require it
    eval::process_access_flags(access_flags, node->get_access_flags(), copy);

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
    const dynd_bool *reduce_axes = rnode->get_reduce_axes();

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

    // Initialize the reduction result.
    // The skip_count is used when there is no reduction identity, and initial values are
    // copied to initialize the result.
    intptr_t skip_count = 0;

    if (rnode->get_identity()) {
        // Copy the identity scalar to the whole result
        intptr_t result_count = 1;
        const intptr_t *result_shape = result->get_shape();
        for (int i = 0, i_end = result->get_ndim(); i != i_end; ++i) {
            result_count *= result_shape[i];
        }
        kernel_instance<unary_operation_pair_t> copy_kernel;
        get_typed_data_assignment_kernel(result_dt, copy_kernel);
        unary_operation_t copy_op = copy_kernel.specializations[scalar_to_contiguous_unary_specialization];
        copy_op(const_cast<char *>(result->get_readonly_originptr()), result_dt.get_data_size(),
                        rnode->get_identity()->get_readonly_originptr(), 0,
                        result_count, copy_kernel.auxdata);
    } else {
        // Copy the first element along each reduction dimension
        kernel_instance<unary_operation_pair_t> copy_kernel;
        if (kernels.empty()) {
            // Straightforward copy kernel
            get_typed_data_assignment_kernel(result_dt, copy_kernel);
        } else {
            // Borrow all the kernels so we can make a copy kernel for this part
            deque<kernel_instance<unary_operation_pair_t>> borrowed_kernels(kernels.size());
            for (size_t i = 0, i_end = kernels.size(); i != i_end; ++i) {
                borrowed_kernels[i].borrow_from(kernels[i]);
            }
            make_buffered_chain_unary_kernel(borrowed_kernels, element_sizes, copy_kernel);
        }

        // Create the shape for the result and
        // the src shape with the first reduce elements cut out
        dimvector result_shape(src_ndim), tmp_src_strides(src_ndim);
        skip_count = 1;
        bool any_reduction = false;
        for (int i = 0; i < src_ndim; ++i) {
            if (reduce_axes[i]) {
                if (src_shape[i] > 1) {
                    any_reduction = true;
                }
                result_shape[i] = 1;
                tmp_src_strides[i] = 0;
            } else {
                skip_count *= src_shape[i];
                result_shape[i] = src_shape[i];
                tmp_src_strides[i] = adjusted_src_strides[i];
            }
        }

        // Copy all the initial elements
        raw_ndarray_iter<1,1> iter(src_ndim, result_shape.get(), result_originptr, result_strides.get(),
                                    src_originptr, tmp_src_strides.get(), axis_perm.get());
        intptr_t innersize = iter.innersize();
        intptr_t dst_stride = iter.innerstride<0>();
        intptr_t src0_stride = iter.innerstride<1>();
        unary_specialization_t uspec = get_unary_specialization(dst_stride, result_dt.get_data_size(),
                                                    src0_stride, strided_node->get_type().storage_type().get_data_size());
        unary_operation_t copy_op = copy_kernel.specializations[uspec];
        if (innersize > 0) {
            do {
                copy_op(iter.data<0>(), dst_stride,
                            iter.data<1>(), src0_stride,
                            innersize, copy_kernel.auxdata);
            } while (iter.iternext());
        }

        // If there isn't any actual reduction going on, return the result of the copy
        if (!any_reduction) {
            return DYND_MOVE(result);
        }
    }

    // Set up the iterator
    raw_ndarray_iter<1,1> iter(src_ndim, src_shape,
                    result_originptr, result_strides.get(),
                    src_originptr, adjusted_src_strides.get());

    // Get the kernel to use in the inner loop
    kernel_instance<unary_operation_t> reduce_operation;
    unary_operation_t reduce_op_duped[4];

    intptr_t innersize = iter.innersize();
    intptr_t dst_stride = iter.innerstride<0>();
    intptr_t src0_stride = iter.innerstride<1>();
    unary_specialization_t uspec = get_unary_specialization(dst_stride, result_dt.get_data_size(),
                                                src0_stride, strided_node->get_type().storage_type().get_data_size());

    // Create the reduction kernel
    rnode->get_unary_operation(dst_stride, src0_stride, reduce_operation);
    if (!kernels.empty()) {
        // Create a unary specialization kernel by replicating the general kernel
        element_sizes.push_back(node->get_type().get_data_size());
        reduce_op_duped[0] = reduce_operation.kernel;
        reduce_op_duped[1] = reduce_operation.kernel;
        reduce_op_duped[2] = reduce_operation.kernel;
        reduce_op_duped[3] = reduce_operation.kernel;
        kernels.push_back(kernel_instance<unary_operation_pair_t>());
        kernels.back().specializations = reduce_op_duped;
        kernels.back().auxdata.swap(reduce_operation.auxdata);

        // Create the chained kernel
        kernel_instance<unary_operation_pair_t> chained_kernel;
        make_buffered_chain_unary_kernel(kernels, element_sizes, chained_kernel);
        // Pick out the right specialization
        reduce_operation.kernel = chained_kernel.specializations[uspec];
        reduce_operation.auxdata.swap(chained_kernel.auxdata);
    }

    if (skip_count > 0) {
        bool more_iteration = false;
        do {
            // This subtracts the number of elements skipped from skip_count
            if (iter.skip_first_visits<0>(skip_count)) {
                reduce_operation.kernel(iter.data<0>() + dst_stride, dst_stride,
                            iter.data<1>() + src0_stride, src0_stride,
                            innersize - 1, reduce_operation.auxdata);
            } else {
                reduce_operation.kernel(iter.data<0>(), dst_stride,
                            iter.data<1>(), src0_stride,
                            innersize, reduce_operation.auxdata);
            }
        } while ((more_iteration = iter.iternext()) && skip_count > 0);
        if (!more_iteration) {
            return DYND_MOVE(result);
        }
    }

    if (innersize > 0) {
        do {
            reduce_operation.kernel(iter.data<0>(), dst_stride,
                        iter.data<1>(), src0_stride,
                        innersize, reduce_operation.auxdata);
        } while (iter.iternext());
    }

    return DYND_MOVE(result);
}

#endif // TODO reenable?
