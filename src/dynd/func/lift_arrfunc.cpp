//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/lift_arrfunc.hpp>
#include <dynd/kernels/make_lifted_ckernel.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>

using namespace std;
using namespace dynd;

namespace {

static void delete_lifted_expr_arrfunc_data(void *self_data_ptr)
{
    memory_block_data *data =
        reinterpret_cast<memory_block_data *>(self_data_ptr);
    memory_block_decref(data);
}

static intptr_t instantiate_lifted_expr_arrfunc_data(
    const arrfunc_type_data *self, dynd::ckernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type *src_tp,
    const char *const *src_arrmeta, uint32_t kernreq,
    const eval::eval_context *ectx)
{
    const array_preamble *data =
        reinterpret_cast<const array_preamble *>(self->data_ptr);
    const arrfunc_type_data *child_af =
        reinterpret_cast<const arrfunc_type_data *>(data->m_data_pointer);
    return make_lifted_expr_ckernel(child_af,
                    ckb, ckb_offset,
                    dst_tp, dst_arrmeta,
                    src_tp, src_arrmeta,
                    static_cast<dynd::kernel_request_t>(kernreq),
                    ectx);
}

} // anonymous namespace

/** Prepends "Dims..." to all the types in the proto */
static ndt::type lift_proto(const ndt::type& proto)
{
    const funcproto_type *p = proto.tcast<funcproto_type>();
    const ndt::type *param_types = p->get_param_types_raw();
    nd::array out_param_types = nd::empty(
        p->get_param_count(), ndt::make_strided_of_type());
    nd::string dimsname("Dims");
    ndt::type *pt = reinterpret_cast<ndt::type *>(
        out_param_types.get_readwrite_originptr());
    for (size_t i = 0, i_end = p->get_param_count(); i != i_end; ++i) {
        pt[i] = ndt::make_ellipsis_dim(dimsname, param_types[i]);
    }
    return ndt::make_funcproto(
        out_param_types,
        ndt::make_ellipsis_dim(dimsname, p->get_return_type()));
}

void dynd::lift_arrfunc(arrfunc_type_data *out_af, const nd::arrfunc &af)
{
    const arrfunc_type_data *af_ptr = af.get();
    if (af_ptr->ckernel_funcproto == unary_operation_funcproto) {
        throw runtime_error("lift_arrfunc() for unary operations is not finished");
    } else if (af_ptr->ckernel_funcproto == expr_operation_funcproto) {
        out_af->free_func = &delete_lifted_expr_arrfunc_data;
        out_af->data_ptr = nd::array(af).release();
        out_af->instantiate = &instantiate_lifted_expr_arrfunc_data;
        out_af->ckernel_funcproto = expr_operation_funcproto;
        out_af->func_proto = lift_proto(af_ptr->func_proto);
    } else {
        stringstream ss;
        ss << "lift_arrfunc() unrecognized ckernel function"
           << " prototype enum value " << af_ptr->ckernel_funcproto;
        throw runtime_error(ss.str());
    }
}
