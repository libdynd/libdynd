//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/ckernel_deferred.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/types/expr_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/type.hpp>

using namespace std;
using namespace dynd;

namespace {

////////////////////////////////////////////////////////////////
// Structure and functions for the unary assignment as a deferred ckernel

struct unary_assignment_ckernel_deferred_data {
    const dynd::base_type *data_types[2];
    assign_error_mode errmode;
    eval::eval_context ectx;
};

static void delete_unary_assignment_ckernel_deferred_data(void *self_data_ptr)
{
    unary_assignment_ckernel_deferred_data *data =
                    reinterpret_cast<unary_assignment_ckernel_deferred_data *>(self_data_ptr);
    base_type_xdecref(data->data_types[0]);
    base_type_xdecref(data->data_types[1]);
    delete data;
}

static void instantiate_unary_assignment_ckernel(void *self_data_ptr,
                dynd::ckernel_builder *out_ckb, size_t ckb_offset,
                const char *const* dynd_metadata, uint32_t kerntype)
{
    unary_assignment_ckernel_deferred_data *data =
                    reinterpret_cast<unary_assignment_ckernel_deferred_data *>(self_data_ptr);
    make_assignment_kernel(out_ckb, ckb_offset,
                    ndt::type(data->data_types[0], true), dynd_metadata[0],
                    ndt::type(data->data_types[1], true), dynd_metadata[1],
                    (kernel_request_t)kerntype, data->errmode, &data->ectx);
}

static void destroy_unary_as_expr_adapter(ckernel_prefix *ckp)
{
    ckernel_prefix *child = ckp + 1;
    if (child->destructor != NULL) {
        child->destructor(child);
    }
}

static void unary_as_expr_adapter_single(
                char *dst, const char * const *src,
                ckernel_prefix *ckp)
{
    ckernel_prefix *child = ckp + 1;
    unary_single_operation_t childop = child->get_function<unary_single_operation_t>();
    childop(dst, *src, child);
}

static void unary_as_expr_adapter_strided(
                char *dst, intptr_t dst_stride,
                const char * const *src, const intptr_t *src_stride,
                size_t count, ckernel_prefix *ckp)
{
    ckernel_prefix *child = ckp + 1;
    unary_strided_operation_t childop = child->get_function<unary_strided_operation_t>();
    childop(dst, dst_stride, *src, *src_stride, count, child);
}

static void instantiate_adapted_expr_assignment_ckernel(void *self_data_ptr,
                dynd::ckernel_builder *out_ckb, size_t ckb_offset,
                const char *const* dynd_metadata, uint32_t kerntype)
{
    unary_assignment_ckernel_deferred_data *data =
                    reinterpret_cast<unary_assignment_ckernel_deferred_data *>(self_data_ptr);
    // Add an adapter kernel which converts the unary kernel to an expr kernel
    out_ckb->ensure_capacity(ckb_offset + sizeof(ckernel_prefix));
    ckernel_prefix *ckp = out_ckb->get_at<ckernel_prefix>(ckb_offset);
    ckp->destructor = &destroy_unary_as_expr_adapter;
    if (kerntype == kernel_request_single) {
        ckp->set_function<expr_single_operation_t>(unary_as_expr_adapter_single);
    } else if (kerntype == kernel_request_strided) {
        ckp->set_function<expr_strided_operation_t>(unary_as_expr_adapter_strided);
    } else {
        throw runtime_error("unsupported kernel request in instantiate_expr_assignment_ckernel");
    }
    make_assignment_kernel(out_ckb, ckb_offset + sizeof(ckernel_prefix),
                    ndt::type(data->data_types[0], true), dynd_metadata[0],
                    ndt::type(data->data_types[1], true), dynd_metadata[1],
                    (kernel_request_t)kerntype, data->errmode, &data->ectx);
}

////////////////////////////////////////////////////////////////
// Structure and functions for the expr as a deferred ckernel

struct expr_ckernel_deferred_data {
    assign_error_mode errmode;
    eval::eval_context ectx;
    const dynd::expr_type *expr_type;
    size_t data_types_size;
    const dynd::base_type *data_types[1];
};

static void delete_expr_ckernel_deferred_data(void *self_data_ptr)
{
    expr_ckernel_deferred_data *data =
                    reinterpret_cast<expr_ckernel_deferred_data *>(self_data_ptr);
    base_type_xdecref(data->expr_type);
    const dynd::base_type **data_types = &data->data_types[0];
    for (size_t i = 0; i < data->data_types_size; ++i) {
        base_type_xdecref(data_types[i]);
    }
    // Call the destructor and free the memory
    data->~expr_ckernel_deferred_data();
    free(data);
}

static void instantiate_expr_ckernel(void *self_data_ptr,
                dynd::ckernel_builder *out_ckb, size_t ckb_offset,
                const char *const* dynd_metadata, uint32_t kerntype)
{
    expr_ckernel_deferred_data *data =
                    reinterpret_cast<expr_ckernel_deferred_data *>(self_data_ptr);
    const expr_kernel_generator& kgen = data->expr_type->get_kgen();
    kgen.make_expr_kernel(out_ckb, ckb_offset,
                    ndt::type(data->data_types[0], true), dynd_metadata[0],
                    data->data_types_size - 1, reinterpret_cast<const ndt::type *>(data->data_types) + 1,
                    const_cast<const char **>(dynd_metadata) + 1, (kernel_request_t)kerntype, &data->ectx);
}



} // anonymous namespace

void dynd::make_ckernel_deferred_from_assignment(const ndt::type& dst_tp, const ndt::type& src_tp,
                deferred_ckernel_funcproto_t funcproto,
                assign_error_mode errmode, ckernel_deferred& out_ckd,
                const dynd::eval::eval_context *ectx)
{
    memset(&out_ckd, 0, sizeof(ckernel_deferred));
    if (funcproto == unary_operation_funcproto) {
        // Since a unary operation was requested, it's a straightforward unary assignment ckernel
        unary_assignment_ckernel_deferred_data *data = new unary_assignment_ckernel_deferred_data;
        out_ckd.data_ptr = data;
        out_ckd.free_func = &delete_unary_assignment_ckernel_deferred_data;
        data->data_types[0] = ndt::type(dst_tp).release();
        data->data_types[1] = ndt::type(src_tp).release();
        data->errmode = errmode;
        data->ectx = *ectx;
        out_ckd.instantiate_func = &instantiate_unary_assignment_ckernel;
        out_ckd.ckernel_funcproto = unary_operation_funcproto;
        out_ckd.data_types_size = 2;
        out_ckd.data_dynd_types = data->data_types;
    } else if (funcproto == expr_operation_funcproto) {
        if (src_tp.get_type_id() == expr_type_id) {
            const expr_type *etp = static_cast<const expr_type *>(src_tp.extended());
            const base_struct_type *operands_type = static_cast<const base_struct_type *>(etp->get_operand_type().extended());
            const ndt::type *operand_types = operands_type->get_field_types();
            // Expose the expr type's expression
            intptr_t nargs = operands_type->get_field_count();
            size_t sizeof_data_mem = sizeof(expr_ckernel_deferred_data) + sizeof(void *) * nargs;
            void *data_mem = malloc(sizeof_data_mem);
            memset(data_mem, 0, sizeof_data_mem);
            expr_ckernel_deferred_data *data = reinterpret_cast<expr_ckernel_deferred_data *>(data_mem);
            out_ckd.data_ptr = data;
            out_ckd.free_func = &delete_expr_ckernel_deferred_data;
            data->data_types_size = nargs + 1;
            const dynd::base_type **data_types_arr = &data->data_types[0];
            data_types_arr[0] = ndt::type(dst_tp).release();
            for (intptr_t i = 0; i < nargs; ++i) {
                // Dereference the pointer type in each field
                const pointer_type *field_ptr_type = static_cast<const pointer_type *>(operand_types[i].extended());
                data_types_arr[i+1] = ndt::type(field_ptr_type->get_target_type()).release();
            }
            data->expr_type = static_cast<const expr_type *>(ndt::type(etp, true).release());
            data->errmode = errmode;
            data->ectx = *ectx;
            out_ckd.instantiate_func = &instantiate_expr_ckernel;
            out_ckd.ckernel_funcproto = expr_operation_funcproto;
            out_ckd.data_types_size = nargs + 1;
            out_ckd.data_dynd_types = data->data_types;
        } else {
            // Adapt the assignment to an expr kernel
            unary_assignment_ckernel_deferred_data *data = new unary_assignment_ckernel_deferred_data;
            out_ckd.data_ptr = data;
            out_ckd.free_func = &delete_unary_assignment_ckernel_deferred_data;
            data->data_types[0] = ndt::type(dst_tp).release();
            data->data_types[1] = ndt::type(src_tp).release();
            data->errmode = errmode;
            data->ectx = *ectx;
            out_ckd.instantiate_func = &instantiate_adapted_expr_assignment_ckernel;
            out_ckd.ckernel_funcproto = expr_operation_funcproto;
            out_ckd.data_types_size = 2;
            out_ckd.data_dynd_types = data->data_types;
        }
    } else {
        stringstream ss;
        ss << "unrecognized ckernel function prototype enum value " << funcproto;
        throw runtime_error(ss.str());
    }
}