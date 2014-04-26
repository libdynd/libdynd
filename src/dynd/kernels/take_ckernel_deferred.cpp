//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/take_ckernel_deferred.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

namespace {
struct masked_take_ck : public kernels::expr_ck<masked_take_ck, 2> {
    ndt::type m_dst_tp;
    const char *m_dst_meta;
    intptr_t m_dim_size, m_src0_stride, m_mask_stride;

    inline void single(char *dst, const char * const *src)
    {
        ckernel_prefix *child = get_child_ckernel();
        unary_strided_operation_t child_fn =
                     child->get_function<unary_strided_operation_t>();
        const char *src0 = src[0];
        const char *mask = src[1];
        intptr_t dim_size = m_dim_size, src0_stride = m_src0_stride,
                 mask_stride = m_mask_stride;
        // Start with the dst matching the dim size. (Maybe better to
        // do smaller? This means no resize required in the loop.)
        ndt::var_dim_element_initialize(m_dst_tp, m_dst_meta, dst, dim_size);
        var_dim_type_data *vdd = reinterpret_cast<var_dim_type_data *>(dst);
        char *dst_ptr = vdd->begin;
        intptr_t dst_stride =
            reinterpret_cast<const var_dim_type_metadata *>(m_dst_meta)->stride;
        intptr_t dst_count = 0;
        intptr_t i = 0;
        while (i < dim_size) {
            // Run of false
            for (; *mask == 0 && i < dim_size;
                 src0 += src0_stride, mask += mask_stride, ++i) {
            }
            // Run of true
            intptr_t i_saved = i;
            for (; *mask != 0 && i < dim_size; mask += mask_stride, ++i) {
            }
            // Copy the run of true
            if (i > i_saved) {
                intptr_t run_count = i - i_saved;
                child_fn(dst_ptr, dst_stride, src0, src0_stride, run_count,
                         child);
                dst_ptr += run_count * dst_stride;
                src0 += run_count * src0_stride;
                dst_count += run_count;
            }
        }
        // Shrink the var dim element to fit
        ndt::var_dim_element_resize(m_dst_tp, m_dst_meta, dst, dst_count);
    }

    inline void destruct_children()
    {
        // The child copy ckernel
        get_child_ckernel()->destroy();
    }
};

struct take_ckernel_deferred_data {
    // The types of the ckernel
    ndt::type data_types[3];
};


static void free_take_ckernel_deferred_data(void *data_ptr) {
    delete reinterpret_cast<take_ckernel_deferred_data *>(data_ptr);
}
} // anonymous namespace

static intptr_t
instantiate_masked_take(void *self_data_ptr, dynd::ckernel_builder *ckb,
            intptr_t ckb_offset, const char *const *dynd_metadata,
            uint32_t kernreq, const eval::eval_context *ectx)
{
    typedef masked_take_ck self_type;

    self_type *self = self_type::create(ckb, ckb_offset, (kernel_request_t)kernreq);
    intptr_t ckb_end = ckb_offset + sizeof(self_type);
    take_ckernel_deferred_data *ckd_data =
        reinterpret_cast<take_ckernel_deferred_data *>(self_data_ptr);

    if (ckd_data->data_types[0].get_type_id() != var_dim_type_id) {
        stringstream ss;
        ss << "take ckernel: could not process type " << ckd_data->data_types[0];
        ss << " as a var dimension";
        throw type_error(ss.str());
    }
    self->m_dst_tp = ckd_data->data_types[0];
    self->m_dst_meta = dynd_metadata[0];
    ndt::type dst_el_tp = self->m_dst_tp.tcast<var_dim_type>()->get_element_type();
    const char *dst_el_meta = self->m_dst_meta + sizeof(var_dim_type_metadata);

    intptr_t src0_dim_size, mask_dim_size;
    ndt::type src0_el_tp, mask_el_tp;
    const char *src0_el_meta, *mask_el_meta;
    if (!ckd_data->data_types[1].get_as_strided_dim(
            dynd_metadata[1], src0_dim_size, self->m_src0_stride, src0_el_tp,
            src0_el_meta)) {
        stringstream ss;
        ss << "take ckernel: could not process type " << ckd_data->data_types[1];
        ss << " as a strided dimension";
        throw type_error(ss.str());
    }
    if (!ckd_data->data_types[2].get_as_strided_dim(
            dynd_metadata[2], mask_dim_size, self->m_mask_stride, mask_el_tp,
            mask_el_meta)) {
        stringstream ss;
        ss << "take ckernel: could not process type " << ckd_data->data_types[2];
        ss << " as a strided dimension";
        throw type_error(ss.str());
    }
    if (src0_dim_size != mask_dim_size) {
        stringstream ss;
        ss << "take ckernel: source data and mask have different sizes, ";
        ss << src0_dim_size << " and " << mask_dim_size;
        throw invalid_argument(ss.str());
    }
    self->m_dim_size = src0_dim_size;
    if (mask_el_tp.get_type_id() != bool_type_id) {
        stringstream ss;
        ss << "take ckernel: mask type should be bool, not ";
        ss << mask_el_tp;
        throw type_error(ss.str());
    }

    // Create the child element assignment ckernel
    return make_assignment_kernel(
        ckb, ckb_end, dst_el_tp, dst_el_meta, src0_el_tp, src0_el_meta,
        kernel_request_strided, assign_error_default, ectx);
}


void kernels::make_take_ckernel_deferred(ckernel_deferred *out_ckd,
                                const ndt::type &dst_tp,
                                const ndt::type &src_tp,
                                const ndt::type &mask_tp)
{
    // Create the data for the ckernel_deferred
    take_ckernel_deferred_data *data = new take_ckernel_deferred_data;
    out_ckd->data_ptr = data;
    out_ckd->free_func = &free_take_ckernel_deferred_data;
    out_ckd->ckernel_funcproto = expr_operation_funcproto;
    out_ckd->data_dynd_types = data->data_types;
    out_ckd->data_types_size = 3;
    out_ckd->instantiate_func = &instantiate_masked_take;
    data->data_types[0] = dst_tp;
    data->data_types[1] = src_tp;
    data->data_types[2] = mask_tp;
}
