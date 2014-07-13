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

    const char *src_it[2] = {src[0], src[1]};
    intptr_t src_it_strides[2] = {m_src_strides[1], 0};
    const strided_dim_type_arrmeta *nh_arrmeta =
        reinterpret_cast<const strided_dim_type_arrmeta *>(m_nh_arrmeta.get());

    // Position the destination at the first output
    dst += m_nh_centre[0] * m_dst_strides[0] +
           m_nh_centre[1] * m_dst_strides[1];
    for (intptr_t coord0 = 0; coord0 < m_shape[0] - nh_arrmeta[0].size;
         ++coord0) {
      // Handle the whole run at once using the child strided ckernel
      nh_op_fn(dst, m_dst_strides[1], src_it, src_it_strides,
               m_shape[1] - nh_arrmeta[1].size, nh_op);
      dst += m_dst_strides[0];
      src_it_strides[0] += m_src_strides[0];
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
  const neighborhood2d *nh = af_self->get_data_as<neighborhood2d>();
  self_type *self = self_type::create(ckb, kernreq, ckb_offset);

  // Process the input for the neighborhood ckernel
  const char *nh_dst_arrmeta = dst_arrmeta;
  ndt::type nh_dst_tp =
      dst_tp.get_type_at_dimension(const_cast<char **>(&nh_dst_arrmeta), 2);
  ndt::type nh_src_tp[2];
  arrmeta_holder(nh_src_tp[0]).swap(self->m_nh_arrmeta);
  const char *nh_src_arrmeta[2] = {self->m_nh_arrmeta.get(), src_arrmeta[1]};
  self->m_nh_shape[0] = nh->nh_shape[0];
  self->m_nh_shape[1] = nh->nh_shape[1];
  self->m_nh_centre[0] = nh->nh_centre[0];
  self->m_nh_centre[1] = nh->nh_centre[1];

  ckb_offset = nh->neighborhood_op.get()->instantiate(
      nh->neighborhood_op.get(), ckb, ckb_offset, nh_dst_tp, nh_dst_arrmeta,
      nh_src_tp, nh_src_arrmeta, kernel_request_strided, ectx);
  return ckb_offset;
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
      "(strided * strided * NH, strided * strided * MSK) -> OUT");
  static ndt::type result_pattern(
      "(strided * strided * NH, strided * strided * MSK) -> strided * strided * OUT");
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
  neighborhood2d *nh = out_af->get_data_as<neighborhood2d>();
  nh->nh_shape[0] = nh_shape[0];
  nh->nh_shape[1] = nh_shape[1];
  nh->nh_centre[0] = nh_centre[0];
  nh->nh_centre[1] = nh_centre[1];
}
