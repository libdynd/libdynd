//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/neighborhood_arrfunc.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/typevar_dim_type.hpp>
#include <dynd/arrmeta_holder.hpp>
#include <dynd/types/type_pattern_match.hpp>
#include <dynd/types/type_substitute.hpp>

using namespace std;
using namespace dynd;

namespace {


class neighborhood2d_ck : public kernels::general_ck<neighborhood2d_ck> {
  // The neighborhood
  intptr_t m_nh_shape[2], m_nh_centre[2];
  // Both src and dst have the same shape
  intptr_t m_shape[2];
  // But may have different strides
  intptr_t m_dst_strides[2], m_src_strides[2];

  static void single(char *dst, const char *const *src, ckernel_prefix *rawself)
  {
    // First pass of this implementation just visits all the complete
    // neighborhoods
    self_type *self = get_self(rawself);
    ckernel_prefix *nh_op = self->get_child_ckernel();
    expr_strided_t nh_op_fn = nh_op->get_function<expr_strided_t>();

    const char *src_it[2] = {src[0], src[1]};
    intptr_t src_it_strides[2] = {self->m_src_strides[1], 0};

    // Position the destination at the first output
    dst += self->m_nh_centre[0] * self->m_dst_strides[0] +
           self->m_nh_centre[1] * self->m_dst_strides[1];
    for (intptr_t coord0 = 0; coord0 < self->m_shape[0] - self->m_nh_shape[0];
         ++coord0) {
      // Handle the whole run at once using the child strided ckernel
      nh_op_fn(dst, self->m_dst_strides[1], src_it, src_it_strides,
               self->m_shape[1] - self->m_nh_shape[1], nh_op);
      dst += self->m_dst_strides[0];
      src_it_strides[0] += self->m_src_strides[0];
    }
  }
};
} // anonymous namespace

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
  

}
