//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/neighborhood_arrfunc.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/arrmeta_holder.hpp>
#include <dynd/types/type_pattern_match.hpp>
#include <dynd/types/type_substitute.hpp>

using namespace std;
using namespace dynd;

namespace {

template <int N>
struct neighborhood_ck;

template <int N, int K>
struct neighborhood_ck_iter;

template <int N>
struct neighborhood_ck_iter<N, N> {
    static void interior(expr_strided_t nh_op_fn,
                         char *dst, intptr_t *dst_strides,
                         const char **src, intptr_t *DYND_UNUSED(src_strides), intptr_t *src_inner_strides,
                         intptr_t *shape, const strided_dim_type_arrmeta *nh_arrmeta, ckernel_prefix *nh_op) {
        nh_op_fn(dst, dst_strides[N], src, src_inner_strides, shape[N] - nh_arrmeta[N].dim_size + 1, nh_op);
    }
};

template <int N, int K = 0>
struct neighborhood_ck_iter {
    static void interior(expr_strided_t nh_op_fn,
                         char *dst, intptr_t *dst_strides,
                         const char **src, intptr_t *src_strides, intptr_t *src_inner_strides,
                         intptr_t *shape, const strided_dim_type_arrmeta *nh_arrmeta, ckernel_prefix *nh_op) {
        for (intptr_t i = 0; i < shape[K] - nh_arrmeta[K].dim_size + 1; ++i) {
            neighborhood_ck_iter<N, K + 1>::interior(nh_op_fn, dst, dst_strides,
                                                     src, src_strides, src_inner_strides,
                                                     shape, nh_arrmeta, nh_op);
            dst += dst_strides[K];
            src[0] += src_strides[K];
        }
        dst -= (shape[K] - nh_arrmeta[K].dim_size + 1) * dst_strides[K];
        src[0] -= (shape[K] - nh_arrmeta[K].dim_size + 1) * src_strides[K];
    }
};

template <int N>
struct neighborhood_ck : public kernels::expr_ck<neighborhood_ck<N>, 2> {
  // The neighborhood
  intptr_t m_nh_centre[N];
  // Arrmeta for a full neighborhood "strided * strided * T"
  arrmeta_holder m_nh_arrmeta;
  // The shape of the src/dst arrays
  intptr_t m_shape[N];
  // The strides of the src/dst arrays may differ
  intptr_t m_dst_strides[N], m_src_strides[N];

  inline void single(char *dst, const char *const *src)
  {
    std::cout << "single called" << std::endl;

    ckernel_prefix *nh_op = kernels::expr_ck<neighborhood_ck<N>, 2>::get_child_ckernel();
    expr_strided_t nh_op_fn = nh_op->get_function<expr_strided_t>();

    const char *src_it[1] = {src[0]};
    const strided_dim_type_arrmeta *nh_arrmeta =
        reinterpret_cast<const strided_dim_type_arrmeta *>(m_nh_arrmeta.get());

    // Position the destination at the first output
    for (intptr_t i = 0; i < N; ++i) {
        dst += m_nh_centre[i] * m_dst_strides[i];
    }

    intptr_t src_inner_strides[1] = {m_src_strides[N - 1]};
    neighborhood_ck_iter<N - 1>::interior(nh_op_fn, dst, m_dst_strides, src_it, m_src_strides, src_inner_strides, m_shape, nh_arrmeta, nh_op);
  }
};

template <int N>
struct neighborhood {
  intptr_t nh_shape[N];
  intptr_t nh_centre[N];
  nd::arrfunc neighborhood_op;
};
} // anonymous namespace

template <int N>
static intptr_t instantiate_neighborhood(
    const arrfunc_type_data *af_self, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
  typedef neighborhood_ck<N> self_type;
  const neighborhood<N> *nh = *af_self->get_data_as<const neighborhood<N> *>();
  self_type *self = self_type::create(ckb, kernreq, ckb_offset);

  // Process the dst array striding/types
  const size_stride_t *dst_shape;
  ndt::type nh_dst_tp;
  const char *nh_dst_arrmeta;
  if (!dst_tp.get_as_strided(dst_arrmeta, N, &dst_shape, &nh_dst_tp,
                             &nh_dst_arrmeta)) {
    stringstream ss;
    ss << "neighborhood arrfunc dst must be a 2D strided array, not " << dst_tp;
    throw invalid_argument(ss.str());
  }
  for (int i = 0; i < N; ++i) {
    self->m_dst_strides[i] = dst_shape[i].stride;
    self->m_shape[i] = dst_shape[i].dim_size;
  }

  // Process the src[0] array striding/type
  const size_stride_t *src0_shape;
  ndt::type src0_el_tp;
  const char *src0_el_arrmeta;
  if (!src_tp[0].get_as_strided(src_arrmeta[0], N, &src0_shape, &src0_el_tp,
                                &src0_el_arrmeta)) {
    stringstream ss;
    ss << "neighborhood arrfunc argument 1 must be a 2D strided array, not "
       << src_tp[0];
    throw invalid_argument(ss.str());
  }

  // Synthesize the arrmeta for the src[0] passed to the neighborhood op
  ndt::type nh_src_tp[1];
  nh_src_tp[0] = ndt::make_strided_dim(src0_el_tp, N);
  arrmeta_holder(nh_src_tp[0]).swap(self->m_nh_arrmeta);
  size_stride_t *nh_src0_arrmeta =
      reinterpret_cast<size_stride_t *>(self->m_nh_arrmeta.get());
  for (int i = 0; i < N; ++i) {
    nh_src0_arrmeta[i].dim_size = nh->nh_shape[i];
    nh_src0_arrmeta[i].stride = src0_shape[i].stride;
  }
  const char *nh_src_arrmeta[1] = {self->m_nh_arrmeta.get()};

  for (int i = 0; i < N; ++i) {
    self->m_src_strides[i] = src0_shape[i].stride;
  }

  // Verify that the src0 and dst shapes match
  for (int i = 0; i < N; ++i) {
    if (self->m_shape[i] != src0_shape[i].dim_size) {
      throw broadcast_error(dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0]);
    }
  }

  for (int i = 0; i < N; ++i) {
    self->m_nh_centre[i] = nh->nh_centre[i];
  }

  // Instantiate the neighborhood op
cout << "instantiate " << nh->neighborhood_op << endl;
cout << "instantiate address " << (void *)nh->neighborhood_op.get()->instantiate << endl;
cout << "nh_src_tp[0] " << nh_src_tp[0] << endl;
cout << "nh_dst_tp " << nh_dst_tp << endl;
  ckb_offset = nh->neighborhood_op.get()->instantiate(
      nh->neighborhood_op.get(), ckb, ckb_offset, nh_dst_tp, nh_dst_arrmeta,
      nh_src_tp, nh_src_arrmeta, kernel_request_strided, ectx);
  return ckb_offset;
}

template <int N>
static void free_neighborhood(arrfunc_type_data *self_af)
{
  neighborhood<N> *nh = *self_af->get_data_as<neighborhood<N> *>();
  delete nh;
}

void dynd::make_neighborhood2d_arrfunc(arrfunc_type_data *out_af,
                                       const nd::arrfunc &neighborhood_op,
                                       intptr_t nh_ndim,
                                       const intptr_t *nh_shape,
                                       const intptr_t *nh_centre)
{
  // neighborhood_op should look like
  // (strided * strided * NH, strided * strided * MSK) -> OUT
  // the resulting arrfunc will look like
  // (strided * strided * NH, strided * strided * MSK) -> strided * strided * OUT
  if (nh_ndim == 2) {
      ndt::type nhop_pattern(
          "(strided * strided * NH) -> OUT");
      ndt::type result_pattern(
          "(strided * strided * NH) -> strided * strided * OUT");
      map<nd::string, ndt::type> typevars;
      if (!ndt::pattern_match(neighborhood_op.get()->func_proto, nhop_pattern,
                               typevars)) {
      stringstream ss;
      ss << "provided neighborhood op proto " << neighborhood_op.get()->func_proto
         << " does not match pattern " << nhop_pattern;
      throw invalid_argument(ss.str());
      }
    out_af->func_proto = ndt::substitute(result_pattern, typevars, true);
    out_af->instantiate = &instantiate_neighborhood<2>;
    out_af->free_func = &free_neighborhood<2>;
    neighborhood<2> **nh = out_af->get_data_as<neighborhood<2> *>();
    *nh = new neighborhood<2>;
    (*nh)->nh_shape[0] = nh_shape[0];
    (*nh)->nh_shape[1] = nh_shape[1];
    (*nh)->nh_centre[0] = nh_centre[0];
    (*nh)->nh_centre[1] = nh_centre[1];
    (*nh)->neighborhood_op = neighborhood_op;
  } else if (nh_ndim == 3) {
      ndt::type nhop_pattern(
          "(strided * strided * strided * NH) -> OUT");
      ndt::type result_pattern(
          "(strided * strided * strided * NH) -> strided * strided * strided * OUT");
      map<nd::string, ndt::type> typevars;
      if (!ndt::pattern_match(neighborhood_op.get()->func_proto, nhop_pattern,
                               typevars)) {
      stringstream ss;
      ss << "provided neighborhood op proto " << neighborhood_op.get()->func_proto
         << " does not match pattern " << nhop_pattern;
      throw invalid_argument(ss.str());
      }
    out_af->func_proto = ndt::substitute(result_pattern, typevars, true);
    out_af->instantiate = &instantiate_neighborhood<3>;
    out_af->free_func = &free_neighborhood<3>;
    neighborhood<3> **nh = out_af->get_data_as<neighborhood<3> *>();
    *nh = new neighborhood<3>;
    (*nh)->nh_shape[0] = nh_shape[0];
    (*nh)->nh_shape[1] = nh_shape[1];
    (*nh)->nh_shape[2] = nh_shape[2];
    (*nh)->nh_centre[0] = nh_centre[0];
    (*nh)->nh_centre[1] = nh_centre[1];
    (*nh)->nh_centre[2] = nh_centre[2];
    (*nh)->neighborhood_op = neighborhood_op;
  }
}
