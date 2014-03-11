//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/lift_reduction_ckernel_deferred.hpp>
#include <dynd/kernels/make_lifted_reduction_ckernel.hpp>

using namespace std;
using namespace dynd;


namespace {

struct lifted_reduction_ckernel_deferred_data {
    // Pointer to the child ckernel_deferred
    const ckernel_deferred *child_elwise_reduction;
    const ckernel_deferred *child_dst_initialization;
    nd::array reduction_identity;
    // Reference to the memory blocks owning them
    memory_block_ptr ref_elwise_reduction;
    memory_block_ptr ref_dst_initialization;
    // The types of the child ckernel and this one
    const ndt::type *child_data_types;
    ndt::type data_types[2];
    intptr_t reduction_ndim;
    bool associative, commutative;
    shortvector<bool> reduction_dimflags;
};

static void delete_lifted_reduction_ckernel_deferred_data(void *self_data_ptr)
{
    lifted_reduction_ckernel_deferred_data *self =
                    reinterpret_cast<lifted_reduction_ckernel_deferred_data *>(self_data_ptr);
    self->~lifted_reduction_ckernel_deferred_data();
    free(self);
}

static intptr_t instantiate_lifted_reduction_ckernel_deferred_data(void *self_data_ptr,
                dynd::ckernel_builder *out_ckb, intptr_t ckb_offset,
                const char *const* dynd_metadata, uint32_t kerntype)
{
    lifted_reduction_ckernel_deferred_data *self =
                    reinterpret_cast<lifted_reduction_ckernel_deferred_data *>(self_data_ptr);
    return make_lifted_reduction_ckernel(
                    self->child_elwise_reduction,
                    self->child_dst_initialization,
                    out_ckb, ckb_offset,
                    self->data_types, dynd_metadata,
                    self->reduction_ndim,
                    self->reduction_dimflags.get(),
                    self->associative, self->commutative,
                    self->reduction_identity,
                    static_cast<dynd::kernel_request_t>(kerntype));
}

} // anonymous namespace

void dynd::lift_reduction_ckernel_deferred(ckernel_deferred *out_ckd,
                const nd::array& elwise_reduction_arr,
                const nd::array& dst_initialization_arr,
                const std::vector<ndt::type>& lifted_types,
                intptr_t reduction_ndim,
                const bool *reduction_dimflags,
                bool associative,
                bool commutative,
                const nd::array& reduction_identity)
{
    // Validate the input elwise_reduction ckernel_deferred
    if (elwise_reduction_arr.is_empty()) {
        throw runtime_error("lift_reduction_ckernel_deferred: 'elwise_reduction' may not be empty");
    }
    if (elwise_reduction_arr.get_type().get_type_id() != ckernel_deferred_type_id) {
        stringstream ss;
        ss << "lift_reduction_ckernel_deferred: 'elwise_reduction' must have type "
           << "ckernel_deferred, not " << elwise_reduction_arr.get_type();
        throw runtime_error(ss.str());
    }
    const ckernel_deferred *elwise_reduction =
                reinterpret_cast<const ckernel_deferred *>(elwise_reduction_arr.get_readonly_originptr());
    if (elwise_reduction->instantiate_func == NULL) {
        throw runtime_error("lift_reduction_ckernel_deferred: 'elwise_reduction' must contain a"
                        " non-null ckernel_deferred object");
    }
    if (elwise_reduction->ckernel_funcproto != unary_operation_funcproto) {
        throw runtime_error("lift_reduction_ckernel_deferred: 'elwise_reduction' must contain a"
                        " unary operation ckernel");
    }

    // Validate the input dst_initialization ckernel_deferred
    const ckernel_deferred *dst_initialization = NULL;
    if (!dst_initialization_arr.is_empty()) {
        if (dst_initialization_arr.get_type().get_type_id() != ckernel_deferred_type_id) {
            stringstream ss;
            ss << "lift_reduction_ckernel_deferred: 'dst_initialization' must have type "
               << "ckernel_deferred, not " << dst_initialization_arr.get_type();
            throw runtime_error(ss.str());
        }
        dst_initialization =
                reinterpret_cast<const ckernel_deferred *>(dst_initialization_arr.get_readonly_originptr());
        if (dst_initialization->instantiate_func == NULL) {
            throw runtime_error("lift_reduction_ckernel_deferred: 'dst_initialization' must contain a"
                            " non-null ckernel_deferred object");
        }
        if (dst_initialization->ckernel_funcproto != unary_operation_funcproto) {
            throw runtime_error("lift_reduction_ckernel_deferred: 'dst_initialization' must contain a"
                            " unary operation ckernel");
        }
    }

    if (lifted_types.size() != 2) {
        throw runtime_error("lift_reduction_ckernel_deferred: 'lifted_types' must have size 2");
    }

    lifted_reduction_ckernel_deferred_data *self = new lifted_reduction_ckernel_deferred_data;
    out_ckd->data_ptr = self;
    out_ckd->free_func = &delete_lifted_reduction_ckernel_deferred_data;
    out_ckd->data_types_size = 2;
    self->child_elwise_reduction = elwise_reduction;
    self->child_dst_initialization = dst_initialization;
    if (!reduction_identity.is_empty()) {
        self->reduction_identity = reduction_identity.eval_immutable();
    }
    self->ref_elwise_reduction = elwise_reduction_arr.get_memblock();
    self->ref_dst_initialization = dst_initialization_arr.get_memblock();
    self->data_types[0] = lifted_types[0];
    self->data_types[1] = lifted_types[1];
    self->child_data_types = elwise_reduction->data_dynd_types;
    self->reduction_ndim = reduction_ndim;
    self->associative = associative;
    self->commutative = commutative;
    self->reduction_dimflags.init(reduction_ndim);
    memcpy(self->reduction_dimflags.get(), reduction_dimflags, sizeof(bool) * reduction_ndim);

    out_ckd->instantiate_func = &instantiate_lifted_reduction_ckernel_deferred_data;
    out_ckd->data_dynd_types = &self->data_types[0];
    out_ckd->ckernel_funcproto = unary_operation_funcproto;
}

    // The types of the child ckernel and this one
    const ndt::type *child_data_types;
    ndt::type data_types[2];
    intptr_t reduction_ndim;
    bool associative, commutative;
    shortvector<bool> reduction_dimflags;
