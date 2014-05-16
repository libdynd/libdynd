//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/rolling_arrfunc.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/arrmeta_holder.hpp>

using namespace std;
using namespace dynd;

namespace {
struct strided_rolling_ck : public kernels::assignment_ck<strided_rolling_ck> {
    intptr_t m_window_size;
    intptr_t m_dim_size, m_dst_stride, m_src_stride;
    size_t m_window_op_offset;
    arrmeta_holder m_src_winop_meta;

    inline void single(char *dst, const char *src)
    {
        ckernel_prefix *nachild = get_child_ckernel();
        ckernel_prefix *wopchild = get_child_ckernel(m_window_op_offset);
        unary_strided_operation_t nachild_fn =
                     nachild->get_function<unary_strided_operation_t>();
        unary_strided_operation_t wopchild_fn =
                     wopchild->get_function<unary_strided_operation_t>();
        // Fill in NA/NaN at the beginning
        if (m_dim_size > 0) {
            nachild_fn(dst, m_dst_stride, NULL, 0,
                       std::min(m_window_size - 1, m_dim_size), nachild);
        }
        // Use stride trickery to do this as one strided call
        if (m_dim_size >= m_window_size) {
            wopchild_fn(dst + m_dst_stride * (m_window_size - 1), m_dst_stride,
                        src, m_src_stride, m_dim_size - m_window_size + 1,
                        wopchild);
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
                     wopchild->get_function<unary_strided_operation_t>();
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

struct rolling_arrfunc_data {
    intptr_t window_size;
    // Pointer to the child arrfunc
    const arrfunc_type_data *window_op_af;
    // Reference to the array containing it
    nd::array window_op_af_arr;
    // The types of the ckernel
    ndt::type data_types[2];
};


static void free_rolling_arrfunc_data(void *data_ptr) {
    delete reinterpret_cast<rolling_arrfunc_data *>(data_ptr);
}
} // anonymous namespace

static intptr_t
instantiate_strided(void *self_data_ptr, dynd::ckernel_builder *ckb,
                    intptr_t ckb_offset, const ndt::type &dst_tp,
                    const char *dst_arrmeta, const ndt::type *src_tp,
                    const char *const *src_arrmeta, uint32_t kernreq,
                    const eval::eval_context *ectx)
{
    typedef strided_rolling_ck self_type;
    rolling_arrfunc_data *af_data =
        reinterpret_cast<rolling_arrfunc_data *>(self_data_ptr);

    self_type *self = self_type::create(ckb, ckb_offset, (kernel_request_t)kernreq);
    intptr_t ckb_end = ckb_offset + sizeof(self_type);
    ndt::type dst_el_tp, src_el_tp;
    const char *dst_el_arrmeta, *src_el_arrmeta;
    if (!dst_tp.get_as_strided_dim(dst_arrmeta, self->m_dim_size,
                                   self->m_dst_stride, dst_el_tp,
                                   dst_el_arrmeta)) {
        stringstream ss;
        ss << "rolling window ckernel: could not process type " << dst_tp;
        ss << " as a strided dimension";
        throw type_error(ss.str());
    }
    intptr_t src_dim_size;
    if (!src_tp[0].get_as_strided_dim(src_arrmeta[0], src_dim_size,
                                      self->m_src_stride, src_el_tp,
                                      src_el_arrmeta)) {
        stringstream ss;
        ss << "rolling window ckernel: could not process type " << src_tp[0];
        ss << " as a strided dimension";
        throw type_error(ss.str());
    }
    if (src_dim_size != self->m_dim_size) {
        stringstream ss;
        ss << "rolling window ckernel: source dimension size " << src_dim_size
           << " for type " << src_tp[0]
           << " does not match dest dimension size " << self->m_dim_size
           << " for type " << dst_tp;
        throw type_error(ss.str());
    }
    self->m_window_size = af_data->window_size;
    // Create the NA-filling child ckernel
    ckb_end = kernels::make_constant_value_assignment_ckernel(
        ckb, ckb_end, dst_el_tp, dst_el_arrmeta,
        numeric_limits<double>::quiet_NaN(), kernel_request_strided, ectx);
    // Re-retrieve the self pointer, because it may be at a new memory location now
    self = ckb->get_at<self_type>(ckb_offset);
    // Create the window op child ckernel
    self->m_window_op_offset = ckb_end;
    // We construct array metadata for the window op ckernel to use,
    // without actually creating an nd::array to hold it.
    arrmeta_holder(ndt::make_strided_dim(src_el_tp))
        .swap(self->m_src_winop_meta);
    self->m_src_winop_meta.get_at<strided_dim_type_metadata>(0)->size =
        self->m_window_size;
    self->m_src_winop_meta.get_at<strided_dim_type_metadata>(0)->stride =
        self->m_src_stride;
    if (src_el_tp.get_metadata_size() > 0) {
        src_el_tp.extended()->metadata_copy_construct(
            self->m_src_winop_meta.get() + sizeof(strided_dim_type_metadata),
            src_el_arrmeta, NULL);
    }
    //const char *window_op_meta[2] = {dst_el_meta, self->m_src_winop_meta.get()};

    // Allow expr ckernels as well as unary via an adapter
    if (af_data->window_op_af->ckernel_funcproto != unary_operation_funcproto) {
        if (af_data->window_op_af->ckernel_funcproto == expr_operation_funcproto) {
            ckb_end = kernels::wrap_expr_as_unary_ckernel(
                ckb, ckb_end, kernel_request_strided);
        } else {
            stringstream ss;
            ss << "rolling window ckernel: invalid funcproto "
               << (arrfunc_proto_t)
                  af_data->window_op_af->ckernel_funcproto
               << " in window_op ckernel";
            throw runtime_error(ss.str());
        }
    }
    const char *src_winop_meta = self->m_src_winop_meta.get();
    return af_data->window_op_af->instantiate_func(
        af_data->window_op_af->data_ptr, ckb, ckb_end, dst_el_tp,
        dst_el_arrmeta, &self->m_src_winop_meta.get_type(),
        &src_winop_meta, kernel_request_strided, ectx);
}

void dynd::make_rolling_arrfunc(arrfunc_type_data *out_af,
                                const ndt::type &dst_tp,
                                const ndt::type &src_tp,
                                const nd::array &window_op,
                                intptr_t window_size)
{
    // Validate the input arrfunc
    if (window_op.get_type().get_type_id() != arrfunc_type_id) {
        stringstream ss;
        ss << "make_rolling_arrfunc() 'window_op' must have type "
           << "arrfunc, not " << window_op.get_type();
        throw runtime_error(ss.str());
    }
    const arrfunc_type_data *window_op_af =
        reinterpret_cast<const arrfunc_type_data *>(
            window_op.get_readonly_originptr());
    if (window_op_af->instantiate_func == NULL) {
        throw runtime_error("make_rolling_arrfunc() 'window_op' must contain "
                            "a non-null arrfunc object");
    }
    if (window_op_af->data_types_size != 2) {
        throw runtime_error("make_rolling_arrfunc() 'window_op' must contain "
                            "a unary arrfunc object");
    }

    // Create the data for the arrfunc
    rolling_arrfunc_data *data = new rolling_arrfunc_data;
    out_af->data_ptr = data;
    out_af->free_func = &free_rolling_arrfunc_data;
    out_af->ckernel_funcproto = unary_operation_funcproto;
    out_af->data_dynd_types = data->data_types;
    out_af->data_types_size = 2;
    if (dst_tp.get_type_id() == var_dim_type_id && src_tp.get_type_id() == var_dim_type_id) {
        //out_af->instantiate_func = &instantiate_var;
        delete data;
        throw runtime_error("TODO: rolling ckernel var");
    } else {
        out_af->instantiate_func = &instantiate_strided;
    }
    data->window_size = window_size;
    data->window_op_af = window_op_af;
    data->window_op_af_arr = window_op;
    data->data_types[0] = dst_tp;
    data->data_types[1] = src_tp;
}
