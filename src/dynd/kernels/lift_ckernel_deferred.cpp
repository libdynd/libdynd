//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/lift_ckernel_deferred.hpp>
#include <dynd/kernels/make_lifted_ckernel.hpp>

using namespace std;
using namespace dynd;

namespace {

struct lifted_expr_ckernel_deferred_data {
    // Pointer to the child ckernel_deferred
    const ckernel_deferred *child_ckd;
    // Reference to the array containing it
    memory_block_data *child_ckd_arr;
    // Number of types
    size_t data_types_size;
    // The types of the child ckernel and this one
    const ndt::type *child_data_types;
    const dynd::base_type *data_types[1];
};

static void delete_lifted_expr_ckernel_deferred_data(void *self_data_ptr)
{
    lifted_expr_ckernel_deferred_data *data =
                    reinterpret_cast<lifted_expr_ckernel_deferred_data *>(self_data_ptr);
    if (data->child_ckd_arr != NULL) {
        memory_block_decref(data->child_ckd_arr);
    }
    const dynd::base_type **data_types = &data->data_types[0];
    for (size_t i = 0; i < data->data_types_size; ++i) {
        base_type_xdecref(data_types[i]);
    }
    free(data);
}

static intptr_t instantiate_lifted_expr_ckernel_deferred_data(void *self_data_ptr,
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                const char *const* dynd_metadata, uint32_t kerntype)
{
    lifted_expr_ckernel_deferred_data *data =
                    reinterpret_cast<lifted_expr_ckernel_deferred_data *>(self_data_ptr);
    return make_lifted_expr_ckernel(data->child_ckd,
                    out_ckb, ckb_offset,
                    reinterpret_cast<const ndt::type *>(data->data_types),
                    dynd_metadata,
                    static_cast<dynd::kernel_request_t>(kerntype));
}

} // anonymous namespace

void dynd::lift_ckernel_deferred(ckernel_deferred *out_ckd,
                const nd::array& ckd_arr,
                const std::vector<ndt::type>& lifted_types)
{
    // Validate the input ckernel_deferred
    if (ckd_arr.get_type().get_type_id() != ckernel_deferred_type_id) {
        stringstream ss;
        ss << "lift_ckernel_deferred() 'ckd' must have type "
           << "ckernel_deferred, not " << ckd_arr.get_type();
        throw runtime_error(ss.str());
    }
    const ckernel_deferred *ckd = reinterpret_cast<const ckernel_deferred *>(ckd_arr.get_readonly_originptr());
    if (ckd->instantiate_func == NULL) {
        throw runtime_error("lift_ckernel_deferred() 'ckd' must contain a"
                        " non-null ckernel_deferred object");
    }
    // Validate that all the types are subarrays as needed for lifting
    intptr_t ntypes = ckd->data_types_size;
    if (ntypes != (intptr_t)lifted_types.size()) {
        stringstream ss;
        ss << "lift_ckernel_deferred() 'lifted_types' list must have "
           << "the same number of types as the input ckernel_deferred "
           << "(" << lifted_types.size() << " vs " << ntypes << ")";
        throw runtime_error(ss.str());
    }
    const ndt::type *ckd_types = ckd->data_dynd_types;
    for (intptr_t i = 0; i < ntypes; ++i) {
        if (!lifted_types[i].is_type_subarray(ckd_types[i])) {
            stringstream ss;
            ss << "lift_ckernel_deferred() 'lifted_types[" << i << "]' value must "
               << "have the corresponding input ckernel_deferred type as a subarray "
               << "(" << ckd_types[i] << " is not a subarray of " << lifted_types[i] << ")";
            throw runtime_error(ss.str());
        }
    }

    if (ckd->ckernel_funcproto == unary_operation_funcproto) {
        throw runtime_error("lift_ckernel_deferred() for unary operations is not finished");
    } else if (ckd->ckernel_funcproto == expr_operation_funcproto) {
        size_t sizeof_data_mem = sizeof(lifted_expr_ckernel_deferred_data) + (sizeof(void *) * (ntypes - 1));
        void *data_mem = malloc(sizeof_data_mem);
        memset(data_mem, 0, sizeof_data_mem);
        lifted_expr_ckernel_deferred_data *data = reinterpret_cast<lifted_expr_ckernel_deferred_data *>(data_mem);
        out_ckd->data_ptr = data;
        out_ckd->free_func = &delete_lifted_expr_ckernel_deferred_data;
        out_ckd->data_types_size = ntypes;
        data->data_types_size = ntypes;
        const dynd::base_type **data_types_arr = &data->data_types[0];
        for (intptr_t i = 0; i < ntypes; ++i) {
            data_types_arr[i] = ndt::type(lifted_types[i]).release();
        }
        data->child_ckd = ckd;
        data->child_ckd_arr = ckd_arr.get_memblock().release();
        out_ckd->instantiate_func = &instantiate_lifted_expr_ckernel_deferred_data;
        out_ckd->data_dynd_types = reinterpret_cast<ndt::type *>(data->data_types);
        out_ckd->ckernel_funcproto = expr_operation_funcproto;
    } else {
        stringstream ss;
        ss << "lift_ckernel_deferred() unrecognized ckernel function"
           << " prototype enum value " << ckd->ckernel_funcproto;
        throw runtime_error(ss.str());
    }
}
