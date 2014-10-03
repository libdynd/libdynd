//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/neighborhood_arrfunc.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/tuple_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/arrmeta_holder.hpp>
#include <dynd/types/type_pattern_match.hpp>
#include <dynd/types/type_substitute.hpp>

using namespace std;
using namespace dynd;

namespace {

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
    intptr_t ndim;
    dimvector nh_shape;
    dimvector nh_offset;
    nd::arrfunc neighborhood_op;
};

} // anonymous namespace


template <int N>
static intptr_t instantiate_neighborhood(
    const arrfunc_type_data *af_self, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const nd::array &aux, const eval::eval_context *ectx)
{
    const neighborhood *nh = *af_self->get_data_as<const neighborhood *>();

    // Process the dst array striding/types
    const size_stride_t *dst_shape;
    ndt::type nh_dst_tp;
    const char *nh_dst_arrmeta;
    if (!dst_tp.get_as_strided(dst_arrmeta, nh->ndim, &dst_shape, &nh_dst_tp,
                               &nh_dst_arrmeta)) {
        stringstream ss;
        ss << "neighborhood arrfunc dst must be a strided array, not " << dst_tp;
        throw invalid_argument(ss.str());
    }

    // Process the src[0] array striding/type
    const size_stride_t *src0_shape;
    ndt::type src0_el_tp;
    const char *src0_el_arrmeta;
    if (!src_tp[0].get_as_strided(src_arrmeta[0], nh->ndim, &src0_shape, &src0_el_tp,
                                  &src0_el_arrmeta)) {
        stringstream ss;
        ss << "neighborhood arrfunc argument 1 must be a 2D strided array, not "
           << src_tp[0];
        throw invalid_argument(ss.str());
    }

    // Synthesize the arrmeta for the src[0] passed to the neighborhood op
    ndt::type nh_src_tp[1];
    nh_src_tp[0] = ndt::make_strided_dim(src0_el_tp, nh->ndim);
    arrmeta_holder nh_arrmeta;
    arrmeta_holder(nh_src_tp[0]).swap(nh_arrmeta);
    size_stride_t *nh_src0_arrmeta = reinterpret_cast<size_stride_t *>(nh_arrmeta.get());
    for (intptr_t i = 0; i < nh->ndim; ++i) {
        nh_src0_arrmeta[i].dim_size = nh->nh_shape[i];
        nh_src0_arrmeta[i].stride = src0_shape[i].stride;
    }
    const char *nh_src_arrmeta[1] = {nh_arrmeta.get()};

    start_stop_t *nh_start_stop = (start_stop_t *) malloc(nh->ndim * sizeof(start_stop_t));

    for (intptr_t i = 0; i < nh->ndim; ++i) {
        typedef neighborhood_ck<N> self_type;
        self_type *self = self_type::create(ckb, kernreq, ckb_offset);

        self->dst_stride = dst_shape[i].stride;
        for (intptr_t j = 0; j < N; ++j) {
            self->src_offset[j] = nh->nh_offset[i] * src0_shape[i].stride;
            self->src_stride[j] = src0_shape[i].stride;
        }

        self->count[0] = -nh->nh_offset[i];
        if (self->count[0] < 0) {
            self->count[0] = 0;
        } else if (self->count[0] > dst_shape[i].dim_size) {
            self->count[0] = dst_shape[i].dim_size;
        }
        self->count[2] = nh->nh_shape[i] + nh->nh_offset[i] - 1;
        if (self->count[2] < 0) {
            self->count[2] = 0;
        } else if (self->count[2] > (dst_shape[i].dim_size - self->count[0])) {
            self->count[2] = dst_shape[i].dim_size - self->count[0];
        }
        self->count[1] = dst_shape[i].dim_size - self->count[0] - self->count[2];

        self->nh_size = nh->nh_shape[i];
        self->nh_start_stop = nh_start_stop + i;
    }

    std::cout << reinterpret_cast<intptr_t>(nh_start_stop) << std::endl;
    ckb_offset = nh->neighborhood_op.get()->instantiate(
        nh->neighborhood_op.get(), ckb, ckb_offset, nh_dst_tp, nh_dst_arrmeta,
        nh_src_tp, nh_src_arrmeta, kernel_request_single, pack(aux, "start_stop", reinterpret_cast<intptr_t>(nh_start_stop)), ectx);
    return ckb_offset;
}

static void free_neighborhood(arrfunc_type_data *self_af) {
    neighborhood *nh = *self_af->get_data_as<neighborhood *>();
    delete nh;
}

inline std::string gen(int n) {
    if (n == 1) {
        return "strided";
    }

    return "strided * " + gen(n - 1);
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

void dynd::make_neighborhood_arrfunc(arrfunc_type_data *out_af, const nd::arrfunc &neighborhood_op,
                                     intptr_t nh_ndim, const intptr_t *nh_shape, const intptr_t *nh_offset)
{
    ndt::type nhop_pattern("(" + gen(nh_ndim) + " * NH) -> OUT");
    ndt::type result_pattern("(" + gen(nh_ndim) + " * NH) -> " + gen(nh_ndim) + " * OUT");

    map<nd::string, ndt::type> typevars;
    if (!ndt::pattern_match(neighborhood_op.get()->func_proto, nhop_pattern, typevars)) {
        stringstream ss;
        ss << "provided neighborhood op proto " << neighborhood_op.get()->func_proto
           << " does not match pattern " << nhop_pattern;
        throw invalid_argument(ss.str());
    }
    out_af->func_proto = ndt::substitute(result_pattern, typevars, true);
    out_af->instantiate = &instantiate_neighborhood<1>;
    out_af->resolve_dst_shape = &resolve_neighborhood_dst_shape;
    out_af->free_func = &free_neighborhood;

    neighborhood **nh = out_af->get_data_as<neighborhood *>();
    *nh = new neighborhood;
    (*nh)->ndim = nh_ndim;
    (*nh)->nh_shape.init(nh_ndim, nh_shape);
    if (nh_offset == NULL) {
        (*nh)->nh_offset.init(nh_ndim);
        for (intptr_t i = 0; i < nh_ndim; ++i) {
            (*nh)->nh_offset[i] = 0;
        }
    } else {
        (*nh)->nh_offset.init(nh_ndim, nh_offset);
    }
    (*nh)->neighborhood_op = neighborhood_op;
}
