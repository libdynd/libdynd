//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/ckernel_deferred.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/type.hpp>

using namespace std;
using namespace dynd;

namespace {

struct unary_assignment_ckernel_deferred_data {
    const dynd::base_type *data_types[2];
    assign_error_mode errmode;
    eval::eval_context ectx;
};

static void delete_unary_assignment_ckernel_deferred_data(void *self_data_ptr)
{
    unary_assignment_ckernel_deferred_data *data =
                    reinterpret_cast<unary_assignment_ckernel_deferred_data *>(self_data_ptr);
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
cout << "ckd " << __LINE__ << endl;
    ckernel_prefix *child = ckp + 1;
    unary_single_operation_t childop = child->get_function<unary_single_operation_t>();
    childop(dst, *src, child);
cout << "ckd " << __LINE__ << endl;
}

static void unary_as_expr_adapter_strided(
                char *dst, intptr_t dst_stride,
                const char * const *src, const intptr_t *src_stride,
                size_t count, ckernel_prefix *ckp)
{
cout << "ckd " << __LINE__ << endl;
cout << "count: " << count << endl;
    ckernel_prefix *child = ckp + 1;
    unary_strided_operation_t childop = child->get_function<unary_strided_operation_t>();
    childop(dst, dst_stride, *src, *src_stride, count, child);
cout << "ckd " << __LINE__ << endl;
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
        data->data_types[0] = ndt::type(dst_tp).release();
        data->data_types[1] = ndt::type(src_tp).release();
        data->errmode = errmode;
        data->ectx = *ectx;
        out_ckd.data_ptr = data;
        out_ckd.free_func = &delete_unary_assignment_ckernel_deferred_data;
        out_ckd.instantiate_func = &instantiate_unary_assignment_ckernel;
        out_ckd.ckernel_funcproto = unary_operation_funcproto;
        out_ckd.data_types_size = 2;
        out_ckd.data_dynd_types = data->data_types;
    } else if (funcproto == expr_operation_funcproto) {
        if (src_tp.get_type_id() == expr_type_id) {
        } else {
            unary_assignment_ckernel_deferred_data *data = new unary_assignment_ckernel_deferred_data;
            data->data_types[0] = ndt::type(dst_tp).release();
            data->data_types[1] = ndt::type(src_tp).release();
            data->errmode = errmode;
            data->ectx = *ectx;
            out_ckd.data_ptr = data;
            out_ckd.free_func = &delete_unary_assignment_ckernel_deferred_data;
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