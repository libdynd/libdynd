//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/take_by_pointer_arrfunc.hpp>
#include <dynd/kernels/expr_kernels.hpp>

using namespace std;
using namespace dynd;

struct take_by_pointer_outer_ck : kernels::expr_ck<take_by_pointer_outer_ck, kernel_request_host, 2> {
    const intptr_t dst_size, dst_stride;
    const intptr_t src1_stride;

    take_by_pointer_outer_ck(intptr_t dst_size, intptr_t dst_stride, intptr_t src1_stride)
      : dst_size(dst_size), dst_stride(dst_stride), src1_stride(src1_stride) {
    }

    void single(char *dst, char **src) {
        ckernel_prefix *child = get_child_ckernel();
        expr_single_t child_fn = child->get_function<expr_single_t>();

        char *src_copy[2] = {src[0], src[1]};
        for (intptr_t i = 0; i < dst_size; ++i) {
            child_fn(dst, src_copy, child);
            dst += dst_stride;
            src_copy[1] += src1_stride;
        }
    }
};

struct take_by_pointer_ck : kernels::expr_ck<take_by_pointer_ck, kernel_request_host, 2> {
    const intptr_t src0_size, src0_stride;
    const intptr_t src1_inner_stride;

    take_by_pointer_ck(intptr_t src0_size, intptr_t src0_stride, intptr_t src1_inner_stride)
      : src0_size(src0_size), src0_stride(src0_stride), src1_inner_stride(src1_inner_stride) {
    }

    void single(char *dst, char **src) {
        ckernel_prefix *child = get_child_ckernel();
        expr_single_t child_fn = child->get_function<expr_single_t>();

        intptr_t i = apply_single_index(*reinterpret_cast<const intptr_t *>(src[1]), src0_size, NULL);
        char *src_copy[2] = {src[0] + i * src0_stride, src[1] + src1_inner_stride};
        child_fn(dst, src_copy, child);
    }
};

static intptr_t instantiate_take_by_pointer(
    const arrfunc_type_data *DYND_UNUSED(af_self),
    const arrfunc_type *DYND_UNUSED(af_tp), dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &DYND_UNUSED(args), const nd::array &DYND_UNUSED(kwds))
{
  intptr_t ndim = src_tp[0].get_ndim();

  ndt::type dst_el_tp;
  const char *dst_el_meta;
  const size_stride_t *dst_size_stride;
  if (!dst_tp.get_as_strided(dst_arrmeta, 1, &dst_size_stride, &dst_el_tp,
                             &dst_el_meta)) {
    stringstream ss;
    ss << "take_by_pointer arrfunc: could not process type " << dst_tp;
    ss << " as a strided dimension";
    throw type_error(ss.str());
  }

  ndt::type src_el_tp[2];
  const char *src_el_meta[2];
  const size_stride_t *src_size_stride[2];
  for (intptr_t i = 0; i < 2; ++i) {
    if (!src_tp[i].get_as_strided(src_arrmeta[i], src_tp[i].get_ndim(),
                                  &src_size_stride[i], &src_el_tp[i],
                                  &src_el_meta[i])) {
      stringstream ss;
      ss << "take_by_pointer arrfunc: could not process type " << src_tp[i];
      ss << " as a strided dimension";
      throw type_error(ss.str());
    }
  }

  take_by_pointer_outer_ck::create(
      ckb, kernreq, ckb_offset, dst_size_stride[0].dim_size,
      dst_size_stride[0].stride, src_size_stride[1][0].stride);

  for (intptr_t i = 0; i < ndim; ++i) {
    take_by_pointer_ck::create(
        ckb, kernel_request_single, ckb_offset, src_size_stride[0][i].dim_size,
        src_size_stride[0][i].stride, src_size_stride[1][1].stride);
  }

  return make_assignment_kernel(ckb, ckb_offset, dst_el_tp, dst_el_meta,
                                src_el_tp[0], src_el_meta[0],
                                kernel_request_single, ectx);
}

static int resolve_take_by_pointer_dst_type(
    const arrfunc_type_data *DYND_UNUSED(af_self), const arrfunc_type *af_tp,
    intptr_t nsrc, const ndt::type *src_tp, int throw_on_error,
    ndt::type &out_dst_tp, const nd::array &DYND_UNUSED(args),
    const nd::array &DYND_UNUSED(kwds))
{
  if (nsrc != 2) {
    if (throw_on_error) {
      stringstream ss;
      ss << "Wrong number of arguments to take arrfunc with prototype ";
      ss << af_tp << ", got " << nsrc << " arguments";
      throw invalid_argument(ss.str());
    }
    else {
      return 0;
    }
  }

  ndt::type idx_el_tp = src_tp[1].get_dtype();
  if (idx_el_tp.get_type_id() != (type_id_t)type_id_of<intptr_t>::value) {
    if (throw_on_error) {
      stringstream ss;
      ss << "take: unsupported type for the index " << idx_el_tp
         << ", need intptr";
      throw invalid_argument(ss.str());
    }
    else {
      return 0;
    }
  }

  out_dst_tp = ndt::make_fixed_dim(src_tp[1].get_dim_size(NULL, NULL),
                                   ndt::make_pointer(src_tp[0].get_dtype()));

  return 1;
}

nd::arrfunc dynd::make_take_by_pointer_arrfunc()
{
  static ndt::type param_types[2] = {ndt::type("M * T"), ndt::type("N * Ix")};
  static ndt::type func_proto =
      ndt::make_funcproto(param_types, ndt::type("R * pointer[T]"));

  nd::array af = nd::empty(func_proto);
  arrfunc_type_data *out_af =
      reinterpret_cast<arrfunc_type_data *>(af.get_readwrite_originptr());
  out_af->instantiate = instantiate_take_by_pointer;
  out_af->resolve_dst_type = resolve_take_by_pointer_dst_type;
  af.flag_as_immutable();
  return af;
}
