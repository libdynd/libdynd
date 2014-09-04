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


struct neighborhood2d_ck : public kernels::expr_ck<neighborhood2d_ck, 2> {
  // The neighborhood
  intptr_t m_nh_centre[2];
  // Arrmeta for a full neighborhood "strided * strided * T"
  arrmeta_holder m_nh_arrmeta;
  // Arrmeta for a partial neighborhood "strided * strided * pointer[T]"
  //TODO arrmeta_holder m_nh_partial_arrmeta
  // The shape of the src/dst arrays
  intptr_t m_shape[2];
  // The strides of the src/dst arrays may differ
  intptr_t m_dst_strides[2], m_src_strides[2];

  inline void single(char *dst, const char *const *src)
  {

    // First pass of this implementation just visits all the complete
    // neighborhoods
    ckernel_prefix *nh_op = get_child_ckernel();
    expr_strided_t nh_op_fn = nh_op->get_function<expr_strided_t>();

    const char *src_it[1] = {src[0]};
    intptr_t src_it_strides[1] = {m_src_strides[1]};
    const strided_dim_type_arrmeta *nh_arrmeta =
        reinterpret_cast<const strided_dim_type_arrmeta *>(m_nh_arrmeta.get());

    // Position the destination at the first output
    dst += m_nh_centre[0] * m_dst_strides[0] +
           m_nh_centre[1] * m_dst_strides[1];
    for (intptr_t coord0 = 0; coord0 < m_shape[0] - nh_arrmeta[0].dim_size + 1;
         ++coord0) {
      // Handle the whole run at once using the child strided ckernel
      nh_op_fn(dst, m_dst_strides[1], src_it, src_it_strides,
               m_shape[1] - nh_arrmeta[1].dim_size + 1, nh_op);
      dst += m_dst_strides[0];
      src_it[0] += m_src_strides[0];
    }
  }
};

struct neighborhood2d {
  intptr_t nh_shape[2];
  intptr_t nh_centre[2];
  nd::arrfunc neighborhood_op;
};
} // anonymous namespace

static intptr_t instantiate_neighborhood2d(
    const arrfunc_type_data *af_self, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx)
{
  typedef neighborhood2d_ck self_type;
  const neighborhood2d *nh = *af_self->get_data_as<const neighborhood2d *>();
  self_type *self = self_type::create(ckb, kernreq, ckb_offset);

  // Process the dst array striding/types
  const size_stride_t *dst_shape;
  ndt::type nh_dst_tp;
  const char *nh_dst_arrmeta;
  if (!dst_tp.get_as_strided(dst_arrmeta, 2, &dst_shape, &nh_dst_tp,
                             &nh_dst_arrmeta)) {
    stringstream ss;
    ss << "neighborhood arrfunc dst must be a 2D strided array, not " << dst_tp;
    throw invalid_argument(ss.str());
  }
  self->m_dst_strides[0] = dst_shape[0].stride;
  self->m_dst_strides[1] = dst_shape[1].stride;
  self->m_shape[0] = dst_shape[0].dim_size;
  self->m_shape[1] = dst_shape[1].dim_size;

  // Process the src[0] array striding/type
  const size_stride_t *src0_shape;
  ndt::type src0_el_tp;
  const char *src0_el_arrmeta;
  if (!src_tp[0].get_as_strided(src_arrmeta[0], 2, &src0_shape, &src0_el_tp,
                                &src0_el_arrmeta)) {
    stringstream ss;
    ss << "neighborhood arrfunc argument 1 must be a 2D strided array, not "
       << src_tp[0];
    throw invalid_argument(ss.str());
  }

  // Synthesize the arrmeta for the src[0] passed to the neighborhood op
  ndt::type nh_src_tp[1];
  nh_src_tp[0] = ndt::make_strided_dim(src0_el_tp, 2);
  arrmeta_holder(nh_src_tp[0]).swap(self->m_nh_arrmeta);
  size_stride_t *nh_src0_arrmeta =
      reinterpret_cast<size_stride_t *>(self->m_nh_arrmeta.get());
  nh_src0_arrmeta[0].dim_size = nh->nh_shape[0];
  nh_src0_arrmeta[0].stride = src0_shape[0].stride;
  nh_src0_arrmeta[1].dim_size = nh->nh_shape[1];
  nh_src0_arrmeta[1].stride = src0_shape[1].stride;
  const char *nh_src_arrmeta[1] = {self->m_nh_arrmeta.get()};

  self->m_src_strides[0] = src0_shape[0].stride;
  self->m_src_strides[1] = src0_shape[1].stride;

  // Verify that the src0 and dst shapes match
  if (self->m_shape[0] != src0_shape[0].dim_size ||
      self->m_shape[1] != src0_shape[1].dim_size) {
    throw broadcast_error(dst_tp, dst_arrmeta, src_tp[0], src_arrmeta[0]);
  }

  self->m_nh_centre[0] = nh->nh_centre[0];
  self->m_nh_centre[1] = nh->nh_centre[1];

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

static void free_neighborhood2d(arrfunc_type_data *self_af)
{
  neighborhood2d *nh = *self_af->get_data_as<neighborhood2d *>();
  delete nh;
}

void dynd::make_neighborhood2d_arrfunc(arrfunc_type_data *out_af,
                                       const nd::arrfunc &neighborhood_op,
                                       const intptr_t *nh_shape,
                                       const intptr_t *nh_centre)
{
  // neighborhood_op should look like
  // (strided * strided * NH, strided * strided * MSK) -> OUT
  // the resulting arrfunc will look like
  // (strided * strided * NH, strided * strided * MSK) -> strided * strided * OUT
  static ndt::type nhop_pattern(
      "(strided * strided * NH) -> OUT");
  static ndt::type result_pattern(
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
  out_af->instantiate = &instantiate_neighborhood2d;
  out_af->free_func = &free_neighborhood2d;
  neighborhood2d **nh = out_af->get_data_as<neighborhood2d *>();
  *nh = new neighborhood2d;
  (*nh)->nh_shape[0] = nh_shape[0];
  (*nh)->nh_shape[1] = nh_shape[1];
  (*nh)->nh_centre[0] = nh_centre[0];
  (*nh)->nh_centre[1] = nh_centre[1];
  (*nh)->neighborhood_op = neighborhood_op;
}
