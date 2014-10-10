//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/shape_tools.hpp>
#include <dynd/func/take_by_pointer_arrfunc.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>
#include <dynd/types/pointer_type.hpp>

using namespace std;
using namespace dynd;

struct take_by_pointer_ck : kernels::expr_ck<take_by_pointer_ck, 2> {
    intptr_t dst_dim_size, src0_dim_size, src1_stride, src1_stride_x;
    intptr_t dst_stride, src0_stride;

    void single(char *dst, const char *const *src) {
        ckernel_prefix *child = get_child_ckernel();
        expr_single_t child_fn = child->get_function<expr_single_t>();

        const char *src0 = src[0];
        const char *src1 = src[1];

        for (intptr_t i = 0; i < dst_dim_size; ++i) {
            intptr_t j = apply_single_index(*reinterpret_cast<const intptr_t *>(src1),
                src0_dim_size, NULL);
            const char *child_src0 = src0 + j * src0_stride;
            const char *child_src[2] = {child_src0, src1 + src1_stride_x};
            child_fn(dst, child_src, child);
            dst += dst_stride;
            src1 += src1_stride;
        }
    }
};

struct take_by_pointer_inner_ck : kernels::expr_ck<take_by_pointer_inner_ck, 2> {
    intptr_t dst_dim_size, src0_dim_size, src1_stride;
    intptr_t dst_stride, src0_stride;

    void single(char *dst, const char *const *src) {
        ckernel_prefix *child = get_child_ckernel();
        expr_single_t child_fn = child->get_function<expr_single_t>();

        const char *src0 = src[0];
        const char *src1 = src[1];

        for (intptr_t i = 0; i < dst_dim_size; ++i) {
            intptr_t j = apply_single_index(*reinterpret_cast<const intptr_t *>(src1),
                src0_dim_size, NULL);
            const char *child_src0 = src0 + j * src0_stride;
            const char *child_pointer_src0 = reinterpret_cast<const char *>(&child_src0);
            child_fn(dst, &child_pointer_src0, child);
            dst += dst_stride;
            src1 += src1_stride;
        }
    }
};

static intptr_t instantiate_apply(const arrfunc_type_data *DYND_UNUSED(af_self),
                                  dynd::ckernel_builder *ckb, intptr_t ckb_offset,
                                  const ndt::type &dst_tp, const char *dst_arrmeta,
                                  const ndt::type *src_tp, const char *const *src_arrmeta,
                                  kernel_request_t kernreq, const nd::array &DYND_UNUSED(aux),
                                  const eval::eval_context *ectx)
{
    intptr_t ndim = dst_tp.get_ndim();

//    ndt::type dst_el_tp, src0_el_tp, src1_el_tp;
  //  const char *dst_el_meta, *src0_el_meta, *src1_el_meta;
    //intptr_t src1_dim_size;

    ndt::type dst_el_tp;
    const char *dst_el_meta;
    const size_stride_t *dst_size_stride;
    if (!dst_tp.get_as_strided(dst_arrmeta, ndim, &dst_size_stride,
                               &dst_el_tp, &dst_el_meta)) {
        stringstream ss;
        ss << "take_by_pointer arrfunc: could not process type " << dst_tp;
        ss << " as a strided dimension";
        throw type_error(ss.str());
    }

    ndt::type src_el_tp[2];
    const char *src_el_meta[2];
    const size_stride_t *src_size_stride[2];
    for (intptr_t i = 0; i < 2; ++i) {
        if (!src_tp[i].get_as_strided(src_arrmeta[i], src_tp[i].get_ndim(), &src_size_stride[i],
                                   &src_el_tp[i], &src_el_meta[i])) {
            stringstream ss;
            ss << "take_by_pointer arrfunc: could not process type " << src_tp[i];
            ss << " as a strided dimension";
            throw type_error(ss.str());
        }
    }

    for (intptr_t i = 0; i < (ndim - 1); ++i) {
        typedef take_by_pointer_ck self_type;
        self_type *self = self_type::create(ckb, kernreq, ckb_offset);

        self->dst_dim_size = dst_size_stride[i].dim_size;
        self->dst_stride = dst_size_stride[i].stride;
        self->src0_dim_size = src_size_stride[0][i].dim_size;
        self->src0_stride = src_size_stride[0][i].stride;
        self->src1_stride = src_size_stride[1][0].stride;
        self->src1_stride_x = src_size_stride[1][1].stride;
    }

    typedef take_by_pointer_inner_ck self_type;
    self_type *self = self_type::create(ckb, kernreq, ckb_offset);

    self->dst_dim_size = dst_size_stride[ndim - 1].dim_size;
    self->dst_stride = dst_size_stride[ndim - 1].stride;
    self->src0_dim_size = src_size_stride[0][ndim - 1].dim_size;
    self->src0_stride = src_size_stride[0][ndim - 1].stride;
    self->src1_stride = src_size_stride[1][0].stride;

    std::cout << "here" << std::endl;

    return make_assignment_kernel(ckb, ckb_offset, dst_el_tp, dst_el_meta,
                                  dst_el_tp, dst_el_meta,
                                  kernel_request_single, ectx);
}

static int resolve_take_dst_type(const arrfunc_type_data *af_self, intptr_t nsrc,
                                 const ndt::type *src_tp,
                                 const nd::array &DYND_UNUSED(dyn_params),
                                 int throw_on_error, ndt::type &out_dst_tp)
{
    if (nsrc != 2) {
      if (throw_on_error) {
        stringstream ss;
        ss << "Wrong number of arguments to take arrfunc with prototype ";
        ss << af_self->func_proto << ", got " << nsrc << " arguments";
        throw invalid_argument(ss.str());
      } else {
        return 0;
      }
    }
    ndt::type mask_el_tp = src_tp[1].get_type_at_dimension(NULL, 1);
    if (mask_el_tp.get_type_id() == bool_type_id) {
        out_dst_tp = ndt::make_var_dim(
            src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type());
    } else if (mask_el_tp.get_type_id() ==
               (type_id_t)type_id_of<intptr_t>::value) {
        if (src_tp[1].get_type_id() == var_dim_type_id) {
            out_dst_tp = ndt::make_var_dim(
                src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type());
        } else {
          out_dst_tp = ndt::make_fixed_dim(
              src_tp[1].get_dim_size(NULL, NULL),
              src_tp[0].get_type_at_dimension(NULL, 1).get_canonical_type());
        }
    } else {
        stringstream ss;
        ss << "take: unsupported type for the index " << mask_el_tp
           << ", need bool or intptr";
        throw invalid_argument(ss.str());
    }

    return 1;
}


static void free(arrfunc_type_data *) {}

void dynd::make_take_by_pointer_arrfunc(arrfunc_type_data *out_af)
{
    static ndt::type param_types[2] = {ndt::type("M * T"), ndt::type("N * Ix")};
    static ndt::type func_proto = ndt::make_funcproto(param_types, ndt::type("R * pointer[T]"));

    out_af->func_proto = func_proto;
    out_af->instantiate = &instantiate_apply;
    out_af->resolve_dst_type = &resolve_take_dst_type;
    out_af->free_func = &free;
}

nd::arrfunc dynd::make_take_by_pointer_arrfunc()
{
    nd::array af = nd::empty(ndt::make_arrfunc());
    make_take_by_pointer_arrfunc(reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr()));
    af.flag_as_immutable();
    return af;
}
