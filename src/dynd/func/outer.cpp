//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arrmeta_holder.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/outer.hpp>
#include <dynd/types/typevar_constructed_type.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::functional::outer(const callable &child)
{
  return callable::make<outer_ck>(outer_make_type(child.get_type()), child);
}

ndt::type nd::functional::outer_make_type(const ndt::callable_type *child_tp)
{
  const std::vector<ndt::type> &param_types = child_tp->get_pos_types();
  std::vector<ndt::type> out_param_types;

  for (intptr_t i = 0, i_end = child_tp->get_npos(); i != i_end; ++i) {
    std::string dimsname("Dims" + std::to_string(i));
    if (param_types[i].get_id() == typevar_constructed_id) {
      out_param_types.push_back(ndt::typevar_constructed_type::make(
          param_types[i].extended<ndt::typevar_constructed_type>()->get_name(),
          ndt::make_ellipsis_dim(dimsname, param_types[i].extended<ndt::typevar_constructed_type>()->get_arg())));
    }
    else {
      out_param_types.push_back(ndt::make_ellipsis_dim(dimsname, param_types[i]));
    }
  }

  ndt::type kwd_tp = child_tp->get_kwd_struct();

  ndt::type ret_tp = child_tp->get_return_type();
  if (ret_tp.get_id() == typevar_constructed_id) {
    ret_tp = ndt::typevar_constructed_type::make(
        ret_tp.extended<ndt::typevar_constructed_type>()->get_name(),
        ndt::make_ellipsis_dim("Dims", ret_tp.extended<ndt::typevar_constructed_type>()->get_arg()));
  }
  else {
    ret_tp = ndt::make_ellipsis_dim("Dims", child_tp->get_return_type());
  }

  return ndt::callable_type::make(ret_tp, ndt::tuple_type::make(out_param_types), kwd_tp);
}
