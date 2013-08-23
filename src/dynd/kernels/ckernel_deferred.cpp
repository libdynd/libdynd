//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/ckernel_deferred.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/type.hpp>

using namespace std;
using namespace dynd;

namespace {

struct unary_assignment_ckernel_deferred_data {
    const dynd::base_type *data_types[2];
    assign_error_mode errmode;
    eval::eval_context ectx;
};

void delete_unary_assignment_ckernel_deferred_data(void *self_data_ptr)
{
    unary_assignment_ckernel_deferred_data *data =
                    reinterpret_cast<unary_assignment_ckernel_deferred_data *>(self_data_ptr);
    delete data;
}

void instantiate_unary_assignment_ckernel_deferred_data(void *self_data_ptr,
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

} // anonymous namespace

void dynd::make_ckernel_deferred_from_assignment(const ndt::type& dst_tp, const ndt::type& src_tp,
                deferred_ckernel_funcproto_t funcproto,
                assign_error_mode errmode, ckernel_deferred& out_ckd,
                const dynd::eval::eval_context *ectx)
{
    memset(&out_ckd, 0, sizeof(ckernel_deferred));
    if (funcproto == unary_operation_funcproto) {
        unary_assignment_ckernel_deferred_data *data = new unary_assignment_ckernel_deferred_data;
        data->data_types[0] = ndt::type(dst_tp).release();
        data->data_types[1] = ndt::type(src_tp).release();
        data->errmode = errmode;
        data->ectx = *ectx;
        out_ckd.data_ptr = data;
        out_ckd.free_func = &delete_unary_assignment_ckernel_deferred_data;
        out_ckd.instantiate_func = &instantiate_unary_assignment_ckernel_deferred_data;
        out_ckd.ckernel_funcproto = unary_operation_funcproto;
        out_ckd.data_types_size = 2;
        out_ckd.data_dynd_types = data->data_types;
    } else if (funcproto == expr_operation_funcproto) {
        out_ckd.ckernel_funcproto = expr_operation_funcproto;
    } else {
        stringstream ss;
        ss << "unrecognized ckernel function prototype enum value " << funcproto;
        throw runtime_error(ss.str());
    }
}