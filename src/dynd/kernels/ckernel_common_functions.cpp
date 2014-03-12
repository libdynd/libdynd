//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>

using namespace std;
using namespace dynd;


void kernels::destroy_trivial_parent_ckernel(ckernel_prefix *ckp)
{
    ckernel_prefix *child = ckp + 1;
    if (child->destructor != NULL) {
        child->destructor(child);
    }
}

void kernels::unary_as_expr_adapter_single_ckernel(
                char *dst, const char * const *src,
                ckernel_prefix *ckp)
{
    ckernel_prefix *child = ckp + 1;
    unary_single_operation_t childop = child->get_function<unary_single_operation_t>();
    childop(dst, *src, child);
}

void kernels::unary_as_expr_adapter_strided_ckernel(
                char *dst, intptr_t dst_stride,
                const char * const *src, const intptr_t *src_stride,
                size_t count, ckernel_prefix *ckp)
{
    ckernel_prefix *child = ckp + 1;
    unary_strided_operation_t childop = child->get_function<unary_strided_operation_t>();
    childop(dst, dst_stride, *src, *src_stride, count, child);
}

intptr_t kernels::wrap_unary_as_expr_ckernel(
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                kernel_request_t kerntype)
{
    // Add an adapter kernel which converts the unary kernel to an expr kernel
    intptr_t ckb_child_offset = ckb_offset + sizeof(ckernel_prefix);
    out_ckb->ensure_capacity(ckb_child_offset);
    ckernel_prefix *ckp = out_ckb->get_at<ckernel_prefix>(ckb_offset);
    ckp->destructor = &kernels::destroy_trivial_parent_ckernel;
    if (kerntype == kernel_request_single) {
        ckp->set_function<expr_single_operation_t>(&kernels::unary_as_expr_adapter_single_ckernel);
    } else if (kerntype == kernel_request_strided) {
        ckp->set_function<expr_strided_operation_t>(&kernels::unary_as_expr_adapter_strided_ckernel);
    } else {
        throw runtime_error("unsupported kernel request in instantiate_expr_assignment_ckernel");
    }
    return ckb_child_offset;
}
                
static void binary_as_unary_right_associative_reduction_adapter_single_ckernel(
                char *dst, const char *src,
                ckernel_prefix *ckp)
{
    // Right associative, evaluate the reduction from right to left:
    //    dst_(0) = a[n-1]
    //    dst_(i+1) = dst_(i) <OP> a[n-1-(i+1)]
    ckernel_prefix *child = ckp + 1;
    expr_single_operation_t childop = child->get_function<expr_single_operation_t>();
    const char *src_binary[2] = {dst, src};
    childop(dst, src_binary, child);
}

static void binary_as_unary_left_associative_reduction_adapter_single_ckernel(
                char *dst, const char *src,
                ckernel_prefix *ckp)
{
    // Left associative, evaluate the reduction from left to right:
    //    dst_(0) = a[0]
    //    dst_(i+1) = a[i+1] <OP> dst_(i)
    ckernel_prefix *child = ckp + 1;
    expr_single_operation_t childop = child->get_function<expr_single_operation_t>();
    const char *src_binary[2] = {src, dst};
    childop(dst, src_binary, child);
}

static void binary_as_unary_right_associative_reduction_adapter_strided_ckernel(
                char *dst, intptr_t dst_stride,
                const char *src, intptr_t src_stride,
                size_t count, ckernel_prefix *ckp)
{
    // Right associative, evaluate the reduction from right to left:
    //    dst_(0) = a[n-1]
    //    dst_(i+1) = dst_(i) <OP> a[n-1-(i+1)]
    ckernel_prefix *child = ckp + 1;
    expr_strided_operation_t childop = child->get_function<expr_strided_operation_t>();
    const char *src_binary[2] = {dst, src};
    const intptr_t src_binary_stride[2] = {dst_stride, src_stride};
    childop(dst, dst_stride, src_binary, src_binary_stride, count, child);
}

static void binary_as_unary_left_associative_reduction_adapter_strided_ckernel(
                char *dst, intptr_t dst_stride,
                const char *src, intptr_t src_stride,
                size_t count, ckernel_prefix *ckp)
{
    // Right associative, evaluate the reduction from right to left:
    //    dst_(0) = a[n-1]
    //    dst_(i+1) = dst_(i) <OP> a[n-1-(i+1)]
    ckernel_prefix *child = ckp + 1;
    expr_strided_operation_t childop = child->get_function<expr_strided_operation_t>();
    const char *src_binary[2] = {src, dst};
    const intptr_t src_binary_stride[2] = {src_stride, dst_stride};
    childop(dst, dst_stride, src_binary, src_binary_stride, count, child);
}

intptr_t kernels::wrap_binary_as_unary_reduction_ckernel(
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                bool right_associative,
                kernel_request_t kerntype)
{
    // Add an adapter kernel which converts the binary expr kernel to an expr kernel
    intptr_t ckb_child_offset = ckb_offset + sizeof(ckernel_prefix);
    out_ckb->ensure_capacity(ckb_child_offset);
    ckernel_prefix *ckp = out_ckb->get_at<ckernel_prefix>(ckb_offset);
    ckp->destructor = &kernels::destroy_trivial_parent_ckernel;
    if (kerntype == kernel_request_single) {
        if (right_associative) {
            ckp->set_function<unary_single_operation_t>(&binary_as_unary_right_associative_reduction_adapter_single_ckernel);
        } else {
            ckp->set_function<unary_single_operation_t>(&binary_as_unary_left_associative_reduction_adapter_single_ckernel);
        }
    } else if (kerntype == kernel_request_strided) {
        if (right_associative) {
            ckp->set_function<unary_strided_operation_t>(&binary_as_unary_right_associative_reduction_adapter_strided_ckernel);
        } else {
            ckp->set_function<unary_strided_operation_t>(&binary_as_unary_left_associative_reduction_adapter_strided_ckernel);
        }
    } else {
        throw runtime_error("unsupported kernel request in wrap_binary_as_unary_reduction_ckernel");
    }
    return ckb_child_offset;
}
