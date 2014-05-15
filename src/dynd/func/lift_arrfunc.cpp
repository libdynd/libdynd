//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/lift_arrfunc.hpp>
#include <dynd/kernels/make_lifted_ckernel.hpp>

using namespace std;
using namespace dynd;

namespace {

struct lifted_expr_arrfunc_data {
    // Pointer to the child arrfunc
    const arrfunc *child_af;
    // Reference to the array containing it
    memory_block_data *child_af_arr;
    // Number of types
    intptr_t data_types_size;
    // The types of the child ckernel and this one
    const ndt::type *child_data_types;
    ndt::type data_types[1];
};

static void delete_lifted_expr_arrfunc_data(void *self_data_ptr)
{
    lifted_expr_arrfunc_data *data =
                    reinterpret_cast<lifted_expr_arrfunc_data *>(self_data_ptr);
    if (data->child_af_arr != NULL) {
        memory_block_decref(data->child_af_arr);
    }
    ndt::type *data_types = &data->data_types[0];
    for (intptr_t i = 0; i < data->data_types_size; ++i) {
        data_types[i] = ndt::type();
    }
    free(data);
}

static intptr_t instantiate_lifted_expr_arrfunc_data(
    void *self_data_ptr, dynd::ckernel_builder *ckb, intptr_t ckb_offset,
    const ndt::type &dst_tp, const char *dst_arrmeta, const ndt::type *src_tp,
    const char *const *src_arrmeta, uint32_t kernreq,
    const eval::eval_context *ectx)
{
    lifted_expr_arrfunc_data *data =
                    reinterpret_cast<lifted_expr_arrfunc_data *>(self_data_ptr);
    return make_lifted_expr_ckernel(data->child_af,
                    ckb, ckb_offset,
                    dst_tp, dst_arrmeta,
                    src_tp, src_arrmeta,
                    static_cast<dynd::kernel_request_t>(kernreq),
                    ectx);
}

} // anonymous namespace

void dynd::lift_arrfunc(arrfunc *out_af, const nd::array &af_arr,
                        const std::vector<ndt::type> &lifted_types)
{
    // Validate the input arrfunc
    if (af_arr.get_type().get_type_id() != arrfunc_type_id) {
        stringstream ss;
        ss << "lift_arrfunc() 'af' must have type "
           << "arrfunc, not " << af_arr.get_type();
        throw runtime_error(ss.str());
    }
    const arrfunc *af = reinterpret_cast<const arrfunc *>(af_arr.get_readonly_originptr());
    if (af->instantiate_func == NULL) {
        throw runtime_error("lift_arrfunc() 'af' must contain a"
                        " non-null arrfunc object");
    }
    // Validate that all the types are subarrays as needed for lifting
    intptr_t ntypes = af->data_types_size;
    if (ntypes != (intptr_t)lifted_types.size()) {
        stringstream ss;
        ss << "lift_arrfunc() 'lifted_types' list must have "
           << "the same number of types as the input arrfunc "
           << "(" << lifted_types.size() << " vs " << ntypes << ")";
        throw runtime_error(ss.str());
    }
    const ndt::type *af_types = af->data_dynd_types;
    for (intptr_t i = 0; i < ntypes; ++i) {
        if (!lifted_types[i].is_type_subarray(af_types[i])) {
            stringstream ss;
            ss << "lift_arrfunc() 'lifted_types[" << i << "]' value must "
               << "have the corresponding input arrfunc type as a subarray "
               << "(" << af_types[i] << " is not a subarray of " << lifted_types[i] << ")";
            throw runtime_error(ss.str());
        }
    }

    if (af->ckernel_funcproto == unary_operation_funcproto) {
        throw runtime_error("lift_arrfunc() for unary operations is not finished");
    } else if (af->ckernel_funcproto == expr_operation_funcproto) {
        size_t sizeof_data_mem = sizeof(lifted_expr_arrfunc_data) + (sizeof(void *) * (ntypes - 1));
        void *data_mem = malloc(sizeof_data_mem);
        memset(data_mem, 0, sizeof_data_mem);
        lifted_expr_arrfunc_data *data = reinterpret_cast<lifted_expr_arrfunc_data *>(data_mem);
        out_af->data_ptr = data;
        out_af->free_func = &delete_lifted_expr_arrfunc_data;
        out_af->data_types_size = ntypes;
        data->data_types_size = ntypes;
        ndt::type *data_types_arr = &data->data_types[0];
        for (intptr_t i = 0; i < ntypes; ++i) {
            data_types_arr[i] = lifted_types[i];
        }
        data->child_af = af;
        data->child_af_arr = af_arr.get_memblock().release();
        out_af->instantiate_func = &instantiate_lifted_expr_arrfunc_data;
        out_af->data_dynd_types = &data->data_types[0];
        out_af->ckernel_funcproto = expr_operation_funcproto;
    } else {
        stringstream ss;
        ss << "lift_arrfunc() unrecognized ckernel function"
           << " prototype enum value " << af->ckernel_funcproto;
        throw runtime_error(ss.str());
    }
}
