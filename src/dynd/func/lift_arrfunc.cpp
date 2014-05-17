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

struct lifted_expr_arrfunc_data {
    // Pointer to the child arrfunc
    const arrfunc_type_data *child_af;
    // Reference to the array containing it
    memory_block_data *child_af_arr;
};

static void delete_lifted_expr_arrfunc_data(void *self_data_ptr)
{
    lifted_expr_arrfunc_data *data =
                    reinterpret_cast<lifted_expr_arrfunc_data *>(self_data_ptr);
    if (data->child_af_arr != NULL) {
        memory_block_decref(data->child_af_arr);
    }
    delete data;
}

static intptr_t instantiate_lifted_expr_arrfunc_data(
    const arrfunc_type_data *self, dynd::ckernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type *src_tp,
    const char *const *src_arrmeta, uint32_t kernreq,
    const eval::eval_context *ectx)
{
    lifted_expr_arrfunc_data *data =
                    reinterpret_cast<lifted_expr_arrfunc_data *>(self->data_ptr);
    return make_lifted_expr_ckernel(data->child_af,
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

void dynd::lift_arrfunc(arrfunc_type_data *out_af, const nd::array &af_arr)
{
    // Validate the input arrfunc
    if (af_arr.get_type().get_type_id() != arrfunc_type_id) {
        stringstream ss;
        ss << "lift_arrfunc() 'af' must have type "
           << "arrfunc, not " << af_arr.get_type();
        throw runtime_error(ss.str());
    }
    const arrfunc_type_data *af = reinterpret_cast<const arrfunc_type_data *>(
        af_arr.get_readonly_originptr());
    if (af->instantiate_func == NULL) {
        throw runtime_error("lift_arrfunc() 'af' must contain a"
                            " non-null arrfunc object");
    }

    if (af->ckernel_funcproto == unary_operation_funcproto) {
        throw runtime_error("lift_arrfunc() for unary operations is not finished");
    } else if (af->ckernel_funcproto == expr_operation_funcproto) {
        lifted_expr_arrfunc_data *data = new lifted_expr_arrfunc_data;
        out_af->data_ptr = data;
        out_af->free_func = &delete_lifted_expr_arrfunc_data;
        data->child_af = af;
        data->child_af_arr = af_arr.get_memblock().release();
        out_af->instantiate_func = &instantiate_lifted_expr_arrfunc_data;
        out_af->ckernel_funcproto = expr_operation_funcproto;
        out_af->func_proto = lift_proto(af->func_proto);
    } else {
        stringstream ss;
        ss << "lift_arrfunc() unrecognized ckernel function"
           << " prototype enum value " << af->ckernel_funcproto;
        throw runtime_error(ss.str());
    }
}
