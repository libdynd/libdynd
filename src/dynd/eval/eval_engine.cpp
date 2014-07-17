//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#if 0 // TODO reenable?

#include <dynd/eval/eval_engine.hpp>
#include <dynd/eval/unary_elwise_eval.hpp>
#include <dynd/eval/elwise_reduce_eval.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/raw_iteration.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/buffered_unary_kernels.hpp>
#include <dynd/memblock/pod_memory_block.hpp>

using namespace std;
using namespace dynd;

static ndarray_node_ptr copy_strided_array(ndarray_node* node, uint32_t access_flags)
{
    int ndim = node->get_ndim();
    const ndt::type& dt = node->get_type();
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
    kernel_instance<unary_operation_pair_t> kernel;
    get_typed_data_assignment_kernel(dt, kernel);
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
        get_unary_specialization(dst_innerstride, dt.get_data_size(), src_innerstride, dt.get_data_size())];
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

void dynd::eval::process_access_flags(uint32_t &dst_access_flags, uint32_t src_access_flags, bool &inout_copy_required)
{
    if (dst_access_flags != 0 && dst_access_flags != src_access_flags) {
        if (dst_access_flags&write_access_flag) {
            // If writable is requested, and src isn't writable, must copy
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

static ndarray_node_ptr evaluate_strided_array_expression_dtype(ndarray_node* node, const eval::eval_context *ectx, bool copy, uint32_t access_flags)
{
    const ndt::type& dt = node->get_type();
    const ndt::type& value_dt = dt.value_type();
    ndarray_node_ptr result;
    //int ndim = node->get_ndim();

    kernel_instance<unary_operation_pair_t> operation;
    get_typed_data_assignment_kernel(value_dt, dt, assign_error_nocheck, ectx, operation);

    return evaluate_strided_with_unary_kernel(node, ectx, copy, access_flags, value_dt, operation);
}

static ndarray_node_ptr evaluate_binary_elwise_array(ndarray_node* node, const eval::eval_context *ectx, bool DYND_UNUSED(copy), uint32_t access_flags)
{
    ndarray_node *op1 = node->get_opnode(0);
    ndarray_node *op2 = node->get_opnode(1);

    // Special case of two strided sub-operands, requiring no intermediate buffers
    if (op1->get_category() == strided_array_node_category &&
                op2->get_category() == strided_array_node_category) {
        ndarray_node_ptr result;
        raw_ndarray_iter<1,2> iter(node->get_ndim(), node->get_shape(),
                                    node->get_type().value_type(), result, access_flags,
                                    op1, op2);
        //iter.debug_print(std::cout);

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

        return DYND_MOVE(result);
    }

    stringstream ss;
    ss << "evaluating this expression graph is not yet supported:\n";
    node->debug_print(ss);
    throw runtime_error(ss.str());
}

ndarray_node_ptr dynd::eval::evaluate(ndarray_node *node, const eval::eval_context *ectx, bool copy, uint32_t access_flags)
{
    if ((access_flags&(immutable_access_flag|write_access_flag)) == (immutable_access_flag|write_access_flag)) {
        throw runtime_error("Cannot create an ndarray which is both writable and immutable");
    }

    switch (node->get_category()) {
        case strided_array_node_category:
            if (node->get_type().get_kind() != expr_kind) {
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
        case groupby_node_category:
            throw std::runtime_error("evaluate is not yet implemented for"
                            " nodes with an groupby_node_category category");
        case arbitrary_node_category:
            throw std::runtime_error("evaluate is not yet implemented for"
                            " nodes with an arbitrary_node_category category");
        case scalar_node_category:
            throw std::runtime_error("evaluate is not yet implemented for"
                            " nodes with an scalar_node_category category");
    }

    stringstream ss;
    ss << "Evaluating the following node is not yet supported:\n";
    node->debug_print(ss);
    throw std::runtime_error(ss.str());
}

#endif // TODO reenable?

