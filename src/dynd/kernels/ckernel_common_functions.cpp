//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/array.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>

using namespace std;
using namespace dynd;


void kernels::destroy_trivial_parent_ckernel(ckernel_prefix *self)
{
    self->destroy_child_ckernel(sizeof(ckernel_prefix));
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

void kernels::expr_as_unary_adapter_single_ckernel(
                char *dst, const char *src,
                ckernel_prefix *ckp)
{
    ckernel_prefix *child = ckp + 1;
    expr_single_operation_t childop = child->get_function<expr_single_operation_t>();
    childop(dst, &src, child);
}

void kernels::expr_as_unary_adapter_strided_ckernel(
                char *dst, intptr_t dst_stride,
                const char *src, intptr_t src_stride,
                size_t count, ckernel_prefix *ckp)
{
    ckernel_prefix *child = ckp + 1;
    expr_strided_operation_t childop = child->get_function<expr_strided_operation_t>();
    childop(dst, dst_stride, &src, &src_stride, count, child);
}

namespace {
    struct constant_value_assignment_ck : public kernels::assignment_ck<constant_value_assignment_ck> {
        // Pointer to the array inside of `constant`
        const char *m_constant_data;
        // Reference which owns the constant value to assign
        nd::array m_constant;

        inline void single(char *dst, const char *DYND_UNUSED(src))
        {
            ckernel_prefix *child = get_child_ckernel();
            unary_single_operation_t child_fn = child->get_function<unary_single_operation_t>();
            child_fn(dst, m_constant_data, child);
        }

        inline void strided(char *dst, intptr_t dst_stride, const char *DYND_UNUSED(src),
                            intptr_t DYND_UNUSED(src_stride), size_t count)
        {
            ckernel_prefix *child = get_child_ckernel();
            unary_strided_operation_t child_fn = child->get_function<unary_strided_operation_t>();
            child_fn(dst, dst_stride, m_constant_data, 0, count, child);
        }

        inline void destruct_children()
        {
            // Destroy the child ckernel
            get_child_ckernel()->destroy();
        }
    };
} // anonymous namespace

size_t kernels::make_constant_value_assignment_ckernel(
    ckernel_builder *out_ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_metadata, const nd::array &constant,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
    typedef constant_value_assignment_ck self_type;
    // Initialize the ckernel
    self_type *self = self_type::create(out_ckb, ckb_offset, kernreq);
    // Store the constant data
    self->m_constant = constant.cast(dst_tp).eval_immutable(ectx);
    self->m_constant_data = self->m_constant.get_readonly_originptr();
    // Create the child assignment ckernel
    return make_assignment_kernel(
        out_ckb, ckb_offset + sizeof(self_type), dst_tp, dst_metadata,
        self->m_constant.get_type(), self->m_constant.get_arrmeta(), kernreq,
        assign_error_default, ectx);
}

intptr_t kernels::wrap_unary_as_expr_ckernel(dynd::ckernel_builder *ckb,
                                             intptr_t ckb_offset,
                                             kernel_request_t kernreq)
{
    // Add an adapter kernel which converts the unary kernel to an expr kernel
    intptr_t ckb_child_offset = ckb_offset + sizeof(ckernel_prefix);
    ckb->ensure_capacity(ckb_child_offset);
    ckernel_prefix *ckp = ckb->get_at<ckernel_prefix>(ckb_offset);
    ckp->destructor = &kernels::destroy_trivial_parent_ckernel;
    ckp->set_expr_function(kernreq,
                            &kernels::unary_as_expr_adapter_single_ckernel,
                            &kernels::unary_as_expr_adapter_strided_ckernel);
    return ckb_child_offset;
}

intptr_t kernels::wrap_expr_as_unary_ckernel(dynd::ckernel_builder *ckb,
                                             intptr_t ckb_offset,
                                             kernel_request_t kernreq)
{
    // Add an adapter kernel which converts the expr kernel to a unary kernel
    intptr_t ckb_child_offset = ckb_offset + sizeof(ckernel_prefix);
    ckb->ensure_capacity(ckb_child_offset);
    ckernel_prefix *ckp = ckb->get_at<ckernel_prefix>(ckb_offset);
    ckp->destructor = &kernels::destroy_trivial_parent_ckernel;
    ckp->set_unary_function(kernreq,
                            &kernels::expr_as_unary_adapter_single_ckernel,
                            &kernels::expr_as_unary_adapter_strided_ckernel);
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
                kernel_request_t kernreq)
{
    // Add an adapter kernel which converts the binary expr kernel to an expr kernel
    intptr_t ckb_child_offset = ckb_offset + sizeof(ckernel_prefix);
    out_ckb->ensure_capacity(ckb_child_offset);
    ckernel_prefix *ckp = out_ckb->get_at<ckernel_prefix>(ckb_offset);
    ckp->destructor = &kernels::destroy_trivial_parent_ckernel;
    if (right_associative) {
        ckp->set_unary_function(
            kernreq,
            &binary_as_unary_right_associative_reduction_adapter_single_ckernel,
            &binary_as_unary_right_associative_reduction_adapter_strided_ckernel);
    } else {
        ckp->set_unary_function(
            kernreq,
            &binary_as_unary_left_associative_reduction_adapter_single_ckernel,
            &binary_as_unary_left_associative_reduction_adapter_strided_ckernel);
    }
    return ckb_child_offset;
}
