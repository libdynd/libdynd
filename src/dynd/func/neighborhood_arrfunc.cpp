//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arrmeta_holder.hpp>
#include <dynd/func/call_callable.hpp>
#include <dynd/func/neighborhood_arrfunc.hpp>
#include <dynd/kernels/expr_kernels.hpp>

using namespace std;
using namespace dynd;

template <int N>
struct neighborhood_ck : kernels::expr_ck<neighborhood_ck<N>, N> {
    typedef neighborhood_ck<N> self_type;

    intptr_t dst_stride;
    intptr_t src_offset[N];
    intptr_t src_stride[N];
    intptr_t count[3];
    intptr_t nh_size;
    start_stop_t *nh_start_stop;

    // local index of first in of bounds element in the neighborhood
    // local index of first out of bounds element in the neighborhood

    inline void single(char *dst, const char *const *src) {
        ckernel_prefix *child = self_type::get_child_ckernel();
        expr_single_t child_fn = child->get_function<expr_single_t>();

        const char *src_copy[N];
        memcpy(src_copy, src, sizeof(src_copy));
        for (intptr_t j = 0; j < N; ++j) {
            src_copy[j] += src_offset[j];
        }

        nh_start_stop->start = count[0];
        nh_start_stop->stop = nh_size; // min(nh_size, dst_size)
        for (intptr_t i = 0; i < count[0]; ++i) {
            child_fn(dst, src_copy, child);
            --(nh_start_stop->start);
            dst += dst_stride;
            for (intptr_t j = 0; j < N; ++j) {
                src_copy[j] += src_stride[j];
            }
        }
      //  *nh_start = 0;
    //    *nh_stop = nh_size;
        for (intptr_t i = 0; i < count[1]; ++i) {
            child_fn(dst, src_copy, child);
            dst += dst_stride;
            for (intptr_t j = 0; j < N; ++j) {
                src_copy[j] += src_stride[j];
            }
        }
  //      *nh_start = 0;
//        *nh_stop = count[2]; // 0 if count[2] > 
        for (intptr_t i = 0; i < count[2]; ++i) {
            --(nh_start_stop->stop);
            child_fn(dst, src_copy, child);
            dst += dst_stride;
            for (intptr_t j = 0; j < N; ++j) {
                src_copy[j] += src_stride[j];
            }
        }
    }
};

struct neighborhood {
    nd::arrfunc op;
    start_stop_t *start_stop;
};

template <int N>
static intptr_t instantiate_neighborhood(
    const arrfunc_type_data *af_self, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const nd::array &kwds, const eval::eval_context *ectx)
{
    neighborhood *nh = *af_self->get_data_as<neighborhood *>();
    nd::arrfunc nh_op = nh->op;

    nd::array shape;
    try {
        shape = kwds.p("shape").f("dereference");
    } catch (...) {
        const nd::array &mask = kwds.p("mask").f("dereference");
        shape = nd::array(mask.get_shape());
    }
    intptr_t ndim = shape.get_dim_size();

    nd::array offset;
    try {
        offset = kwds.p("offset").f("dereference");
    } catch (...) {
    }

    // Process the dst array striding/types
    const size_stride_t *dst_shape;
    ndt::type nh_dst_tp;
    const char *nh_dst_arrmeta;
    if (!dst_tp.get_as_strided(dst_arrmeta, ndim, &dst_shape, &nh_dst_tp,
                               &nh_dst_arrmeta)) {
        stringstream ss;
        ss << "neighborhood arrfunc dst must be a strided array, not " << dst_tp;
        throw invalid_argument(ss.str());
    }

    // Process the src[0] array striding/type
    const size_stride_t *src0_shape;
    ndt::type src0_el_tp;
    const char *src0_el_arrmeta;
    if (!src_tp[0].get_as_strided(src_arrmeta[0], ndim, &src0_shape, &src0_el_tp,
                                  &src0_el_arrmeta)) {
        stringstream ss;
        ss << "neighborhood arrfunc argument 1 must be a 2D strided array, not "
           << src_tp[0];
        throw invalid_argument(ss.str());
    }

    // Synthesize the arrmeta for the src[0] passed to the neighborhood op
    ndt::type nh_src_tp[1];
    nh_src_tp[0] = ndt::make_strided_dim(src0_el_tp, ndim);
    arrmeta_holder nh_arrmeta;
    arrmeta_holder(nh_src_tp[0]).swap(nh_arrmeta);
    size_stride_t *nh_src0_arrmeta = reinterpret_cast<size_stride_t *>(nh_arrmeta.get());
    for (intptr_t i = 0; i < ndim; ++i) {
        nh_src0_arrmeta[i].dim_size = shape(i).as<intptr_t>();
        nh_src0_arrmeta[i].stride = src0_shape[i].stride;
    }
    const char *nh_src_arrmeta[1] = {nh_arrmeta.get()};

    start_stop_t *nh_start_stop = (start_stop_t *) malloc(ndim * sizeof(start_stop_t));
    nh->start_stop = nh_start_stop;

    for (intptr_t i = 0; i < ndim; ++i) {
        typedef neighborhood_ck<N> self_type;
        self_type *self = self_type::create(ckb, kernreq, ckb_offset);

        self->dst_stride = dst_shape[i].stride;
        for (intptr_t j = 0; j < N; ++j) {
            self->src_offset[j] = offset.is_null() ? 0 : (offset(i).as<intptr_t>() * src0_shape[i].stride);
            self->src_stride[j] = src0_shape[i].stride;
        }

        self->count[0] = offset.is_null() ? 0 : -offset(i).as<intptr_t>();
        if (self->count[0] < 0) {
            self->count[0] = 0;
        } else if (self->count[0] > dst_shape[i].dim_size) {
            self->count[0] = dst_shape[i].dim_size;
        }
        self->count[2] = shape(i).as<intptr_t>() + (offset.is_null() ? 0 : offset(i).as<intptr_t>()) - 1;
        if (self->count[2] < 0) {
            self->count[2] = 0;
        } else if (self->count[2] > (dst_shape[i].dim_size - self->count[0])) {
            self->count[2] = dst_shape[i].dim_size - self->count[0];
        }
        self->count[1] = dst_shape[i].dim_size - self->count[0] - self->count[2];

        self->nh_size = shape(i).as<intptr_t>();
        self->nh_start_stop = nh_start_stop + i;
    }

    ckb_offset = nh_op.get()->instantiate(nh_op.get(), ckb, ckb_offset,
        nh_dst_tp, nh_dst_arrmeta, nh_src_tp, nh_src_arrmeta,
        kernel_request_single, pack(kwds, "start_stop", reinterpret_cast<intptr_t>(nh_start_stop)), ectx);
    return ckb_offset;
}

static void resolve_neighborhood_dst_shape(const arrfunc_type_data *self,
                                           intptr_t *out_shape,
                                           const ndt::type &dst_tp,
                                           const ndt::type *src_tp,
                                           const char *const *src_arrmeta,
                                           const char *const *src_data) {
    intptr_t param_count = self->get_param_count();
    intptr_t child_ndim = 0;
    intptr_t ndim = dst_tp.get_ndim() - 0 * child_ndim;
    if (ndim > 0) {
        for (intptr_t i = 0; i < param_count; ++i) {
            intptr_t ndim_i = src_tp[i].get_ndim();
            if (ndim_i > 0) {
                src_tp[i].extended()->get_shape(ndim_i, 0, out_shape,
                                                src_arrmeta[i], src_data[i]);
            }
        }
    }
}

static void free_neighborhood(arrfunc_type_data *self_af) {
    neighborhood *nh = *self_af->get_data_as<neighborhood *>();
    free(nh->start_stop);
    delete nh;
}

void dynd::make_neighborhood_arrfunc(arrfunc_type_data *out_af, const nd::arrfunc &neighborhood_op, intptr_t nh_ndim)
{
    std::ostringstream oss;
    oss << "strided**" << nh_ndim;
    ndt::type nhop_pattern("(" + oss.str() + " * NH) -> OUT");
    ndt::type result_pattern("(" + oss.str() + " * NH) -> " + oss.str() + " * OUT");

    map<nd::string, ndt::type> typevars;
    if (!ndt::pattern_match(neighborhood_op.get()->func_proto, nhop_pattern, typevars)) {
        stringstream ss;
        ss << "provided neighborhood op proto " << neighborhood_op.get()->func_proto
           << " does not match pattern " << nhop_pattern;
        throw invalid_argument(ss.str());
    }

    neighborhood **nh = out_af->get_data_as<neighborhood *>();
    *nh = new neighborhood;
    (*nh)->op = neighborhood_op;
    out_af->func_proto = ndt::substitute(result_pattern, typevars, true);
    out_af->instantiate = &instantiate_neighborhood<1>;
    out_af->resolve_dst_shape = &resolve_neighborhood_dst_shape;
    out_af->free_func = &free_neighborhood;
}
