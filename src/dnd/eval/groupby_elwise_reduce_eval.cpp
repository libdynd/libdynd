//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dnd/shape_tools.hpp>
#include <dnd/dtypes/categorical_dtype.hpp>
#include <dnd/eval/groupby_elwise_reduce_eval.hpp>
#include <dnd/eval/unary_elwise_eval.hpp>
#include <dnd/nodes/elwise_reduce_kernel_node.hpp>
#include <dnd/nodes/groupby_node.hpp>

using namespace std;
using namespace dnd;

ndarray_node_ptr dnd::eval::evaluate_groupby_elwise_reduce(ndarray_node *node, const eval::eval_context *ectx,
                                bool copy, uint32_t access_flags)
{
    elwise_reduce_kernel_node *rnode = static_cast<elwise_reduce_kernel_node*>(node);
    groupby_node *gnode = static_cast<groupby_node*>(rnode->get_opnode(0));
    ndarray_node *data_strided_node = gnode->get_data_node();
    ndarray_node *by_strided_node = gnode->get_by_node();

    const dtype& result_dt = rnode->get_dtype().value_dtype();
    const dtype& groups_dt = gnode->get_groups();
    const categorical_dtype *groups = static_cast<const categorical_dtype *>(groups_dt.extended());
    intptr_t num_groups = groups->get_category_count();

    // The structure of this node is (node == elwise_reduce_kernel_node) -> (groupby_node) -> (...)
    // The groupby itself isn't stored, but rather the reduction is
    // accumulated while iterating through the groupby data.

    if (result_dt.get_memory_management() == blockref_memory_management) {
        throw runtime_error("blockref memory management isn't supported for elwise reduce gfuncs yet");
    }

    if (rnode->get_reduce_axes()[0] != false || rnode->get_reduce_axes()[1] != true) {
        throw runtime_error("a groupby reduce can only be done with axis=1");
    }

    // Used when the input is some kind of expression
    deque<unary_specialization_kernel_instance> data_kernels, by_kernels;
    deque<intptr_t> data_element_sizes, by_element_sizes;

    if (data_strided_node->get_category() != strided_array_node_category ||
                    data_strided_node->get_dtype().kind() == expression_kind) {
        data_strided_node = push_front_node_unary_kernels(data_strided_node, ectx, data_kernels, data_element_sizes);
    }

    if (by_strided_node->get_category() != strided_array_node_category ||
                    by_strided_node->get_dtype().kind() == expression_kind) {
        by_strided_node = push_front_node_unary_kernels(by_strided_node, ectx, by_kernels, by_element_sizes);
    }

    // Adjust the access flags, and force a copy if the access flags require it
    eval::process_access_flags(access_flags, node->get_access_flags(), copy);

    // Validate the input shapes of 'data' and 'by'
    if (data_strided_node->get_ndim() != 1 || by_strided_node->get_ndim() != 1) {
        throw runtime_error("groupby reduce only supports one-dimensional inputs presently");
    }
    if (data_strided_node->get_shape()[0] != by_strided_node->get_shape()[0]) {
        stringstream ss;
        ss << "groupby reduce argument shapes, 'data' ";
        print_shape(ss, data_strided_node->get_ndim(), data_strided_node->get_shape());
        ss << " and 'by' ";
        print_shape(ss, by_strided_node->get_ndim(), by_strided_node->get_shape());
        ss << " must be equal";
        throw runtime_error(ss.str());
    }

    // Allocate the memoryblock for the data
    char *result_originptr = NULL;
    intptr_t result_stride = result_dt.element_size();
    memory_block_ptr result_memblock = make_fixed_size_pod_memory_block(result_stride * num_groups,
                    result_dt.alignment(), &result_originptr,
                    NULL, NULL);
    ndarray_node_ptr result;
    result = make_strided_ndarray_node(result_dt, 1,
                &num_groups, &result_stride, result_originptr, access_flags, result_memblock);

    // Get the raw looping variables
    const char *data_ptr = data_strided_node->get_readonly_originptr();
    intptr_t data_stride = data_strided_node->get_strides()[0];
    const char *by_ptr = by_strided_node->get_readonly_originptr();
    intptr_t by_stride = by_strided_node->get_strides()[0];
    intptr_t size = by_strided_node->get_shape()[0];
    
    // If we're doing a right associative reduce, reverse things to process from right to left
    if (rnode->get_rightassoc()) {
        data_ptr += (size - 1) * data_stride;
        data_stride = -data_stride;
        by_ptr += (size - 1) * by_stride;
        by_stride = -by_stride;
    }

    kernel_instance<unary_operation_t> reduce_operation;
    rnode->get_unary_operation(0, 0, reduce_operation);

    if (rnode->get_identity() != NULL) {
        // Fill the result with the identity
        dtype_strided_assign(result_dt, result_originptr, result_dt.element_size(),
                            result_dt, rnode->get_identity()->get_readonly_originptr(), 0,
                            num_groups, assign_error_none, ectx);

        if (data_kernels.empty() && by_kernels.empty()) {
            for (intptr_t i = 0; i < size; ++i) {
                int32_t group = *reinterpret_cast<const int32_t *>(by_ptr);
                if ((uint32_t)group < (uint32_t)num_groups) {
                    reduce_operation.kernel(result_originptr + result_stride * group, 0,
                                    data_ptr, 0, 1, reduce_operation.auxdata);
                } else {
                    stringstream ss;
                    ss << "invalid value " << group << " for categorical dtype " << groups_dt;
                    throw runtime_error(ss.str());
                }
                by_ptr += by_stride;
                data_ptr += data_stride;
            }
        } else {
            throw runtime_error("groupby reduction with expression inputs isn't implemented yet");
        }
    } else {
        // TODO: Make a boolean array to track which elements have been started
        throw runtime_error("groupby reduction with no identity isn't implemented yet");
    }

    return DND_MOVE(result);
}
