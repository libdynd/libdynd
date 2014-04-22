//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/rolling_ckernel_deferred.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/types/var_dim_type.hpp>

using namespace std;
using namespace dynd;

namespace {
struct strided_rolling_ck : public kernels::assignment_ck<strided_rolling_ck> {
    intptr_t m_window_size;
    intptr_t m_dim_size, m_dst_stride, m_src_stride;
    size_t m_window_op_offset;

    inline void single(char *dst, const char *src)
    {
        ckernel_prefix *nachild = get_child_ckernel();
        ckernel_prefix *wopchild = get_child_ckernel(m_window_op_offset);
        unary_strided_operation_t nachild_fn =
                     nachild->get_function<unary_strided_operation_t>();
        unary_strided_operation_t wopchild_fn =
                     nachild->get_function<unary_strided_operation_t>();
        // Fill in NA/NaN at the beginning
        if (m_dim_size > 0) {
            nachild_fn(dst, m_dst_stride, NULL, 0,
                       std::min(m_window_size - 1, m_dim_size), nachild);
        }
        // Use stride trickery to do this as one strided call
        if (m_dim_size >= m_window_size) {
            wopchild_fn(dst + m_dst_stride * (m_window_size - 1), m_dst_stride, src,
                        m_src_stride, m_dim_size - m_window_size + 1, wopchild);
        }
    }

    inline void destruct_children()
    {
        // The window op
        base.destroy_child_ckernel(m_window_op_offset);
        // The NA filler
        base.destroy_child_ckernel(sizeof(self_type));
    }
};

struct var_rolling_ck : public kernels::assignment_ck<strided_rolling_ck> {
    intptr_t m_window_size;
    intptr_t m_src_stride, m_src_offset;
    ndt::type m_dst_tp;
    const char *m_dst_meta;
    size_t m_window_op_offset;

    inline void single(char *dst, const char *src)
    {
        // Get the child ckernels
        ckernel_prefix *nachild = get_child_ckernel();
        ckernel_prefix *wopchild = get_child_ckernel(m_window_op_offset);
        unary_strided_operation_t nachild_fn =
                     nachild->get_function<unary_strided_operation_t>();
        unary_strided_operation_t wopchild_fn =
                     nachild->get_function<unary_strided_operation_t>();
        // Get pointers to the src and dst data
        var_dim_type_data *dst_dat = reinterpret_cast<var_dim_type_data *>(dst);
        intptr_t dst_stride =
            reinterpret_cast<const var_dim_type_metadata *>(m_dst_meta)->stride;
        const var_dim_type_data *src_dat =
            reinterpret_cast<const var_dim_type_data *>(src);
        const char *src_arr_ptr = src_dat->begin + m_src_offset;
        intptr_t dim_size = src_dat->size;
        // Allocate the output data
        ndt::var_dim_element_initialize(m_dst_tp, m_dst_meta, dst, dim_size);
        char *dst_arr_ptr = dst_dat->begin;

        // Fill in NA/NaN at the beginning
        if (dim_size > 0) {
            nachild_fn(dst_arr_ptr, dst_stride, NULL, 0,
                       std::min(m_window_size - 1, dim_size), nachild);
        }
        // Use stride trickery to do this as one strided call
        if (dim_size >= m_window_size) {
            wopchild_fn(dst_arr_ptr + dst_stride * (m_window_size - 1),
                        dst_stride, src_arr_ptr, m_src_stride,
                        dim_size - m_window_size + 1, wopchild);
        }
    }

    inline void destruct_children()
    {
        // The window op
        base.destroy_child_ckernel(m_window_op_offset);
        // The NA filler
        base.destroy_child_ckernel(sizeof(self_type));
    }
};

struct rolling_ckernel_deferred_data {
    intptr_t window_size;
    // Pointer to the child ckernel_deferred
    const ckernel_deferred *window_op_ckd;
    // Reference to the array containing it
    memory_block_data *window_op_ckd_arr;
    // The types of the child ckernel and this one
    const ndt::type *window_op_data_types;
    ndt::type data_types[2];
};

} // anonymous namespace


static intptr_t
instantiate_strided(void *self_data_ptr, dynd::ckernel_builder *ckb,
            intptr_t ckb_offset, const char *const *dynd_metadata,
            uint32_t kernreq, const eval::eval_context *ectx)
{
    typedef strided_rolling_ck self_type;
    rolling_ckernel_deferred_data *ckd_data =
        reinterpret_cast<rolling_ckernel_deferred_data *>(self_data_ptr);

    self_type *self = self_type::create(ckb, ckb_offset, (kernel_request_t)kernreq);
    intptr_t ckb_end = ckb_offset + sizeof(self_type);
    ndt::type dst_el_tp, src_el_tp;
    const char *dst_el_meta, *src_el_meta;
    if (!ckd_data->data_types[0].get_as_strided_dim(
            dynd_metadata[0], self->m_dim_size, self->m_dst_stride, dst_el_tp,
            dst_el_meta)) {
        stringstream ss;
        ss << "rolling window ckernel: could not process type " << ckd_data->data_types[0];
        ss << " as a strided dimension";
        throw type_error(ss.str());
    }
    intptr_t src_dim_size;
    if (!ckd_data->data_types[1].get_as_strided_dim(
            dynd_metadata[1], src_dim_size, self->m_src_stride, src_el_tp,
            src_el_meta)) {
        stringstream ss;
        ss << "rolling window ckernel: could not process type " << ckd_data->data_types[0];
        ss << " as a strided dimension";
        throw type_error(ss.str());
    }
    if (src_dim_size != self->m_dim_size) {
        stringstream ss;
        ss << "rolling window ckernel: source dimension size " << src_dim_size
           << " for type " << ckd_data->data_types[1]
           << " does not match dest dimension size " << self->m_dim_size
           << " for type " << ckd_data->data_types[0];
        throw type_error(ss.str());
    }
    self->m_window_size = ckd_data->window_size;
    // Create the NA-filling child ckernel
    ckb_end = kernels::make_constant_value_assignment_ckernel(
        ckb, ckb_end, dst_el_tp, dst_el_meta,
        numeric_limits<double>::quiet_NaN(), kernel_request_strided, ectx);
    // Re-retrieve the self pointer, because it may be at a new memory location now
    self = ckb->get_at<self_type>(ckb_offset);
    // Create the window op child ckernel
    self->m_window_op_offset = ckb_end;
    ckd_data->window_op_ckd->instantiate_func(ckd_data->window_op_ckd->data_ptr, ckb, ckb_end, )
}

void make_rolling_ckernel_deferred(ckernel_deferred *out_ckd,
                                   const ndt::type &dst_tp,
                                   const ndt::type &src_tp,
                                   const nd::array &window_op, int window_size)
{
}