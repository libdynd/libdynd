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
