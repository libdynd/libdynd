//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arrmeta_holder.hpp>
#include <dynd/gfunc/call_callable.hpp>
#include <dynd/func/neighborhood.hpp>
#include <dynd/kernels/neighborhood.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::functional::neighborhood(const nd::callable &neighborhood_op,
                                          intptr_t nh_ndim)
{
  const ndt::callable_type *funcproto_tp =
      neighborhood_op.get_array_type().extended<ndt::callable_type>();

  nd::array arg_tp = nd::empty(3, ndt::make_type());
  arg_tp(0).vals() = ndt::type("?" + std::to_string(nh_ndim) + " * int");
  arg_tp(1).vals() = ndt::type("?" + std::to_string(nh_ndim) + " * int");
  arg_tp(2).vals() =
      ndt::type("?Fixed**" + std::to_string(nh_ndim) + " * bool");
  std::vector<std::string> arg_names;
  arg_names.push_back("shape");
  arg_names.push_back("offset");
  arg_names.push_back("mask");
  ndt::type ret_tp = funcproto_tp->get_pos_type(0)
                         .with_replaced_dtype(funcproto_tp->get_return_type());
  ndt::type self_tp =
      ndt::callable_type::make(ret_tp, funcproto_tp->get_pos_tuple(),
                               ndt::struct_type::make(arg_names, arg_tp));

  std::ostringstream oss;
  oss << "Fixed**" << nh_ndim;
  ndt::type nhop_pattern("(" + oss.str() + " * NH) -> OUT");
  ndt::type result_pattern("(" + oss.str() + " * NH) -> " + oss.str() +
                           " * OUT");

  map<std::string, ndt::type> typevars;
  if (!nhop_pattern.match(neighborhood_op.get_array_type(), typevars)) {
    stringstream ss;
    ss << "provided neighborhood op proto " << neighborhood_op.get_array_type()
       << " does not match pattern " << nhop_pattern;
    throw invalid_argument(ss.str());
  }

  std::shared_ptr<neighborhood_data> nh(
      new nd::functional::neighborhood_data(neighborhood_op, nh_ndim));
  return callable::make<neighborhood_ck<1>>(self_tp, nh, 0);
}
