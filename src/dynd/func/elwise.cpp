//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/elwise.hpp>
#include <dynd/kernels/elwise.hpp>
#include <dynd/types/dim_fragment_type.hpp>
#include <dynd/types/typevar_constructed_type.hpp>

using namespace std;
using namespace dynd;

ndt::type nd::functional::elwise_make_type(const ndt::callable_type *child_tp)
{
  const std::vector<ndt::type> &param_types = child_tp->get_pos_types();
  std::vector<ndt::type> out_param_types;
  std::string dimsname("Dims");

  for (const ndt::type &t : param_types) {
    if (t.get_id() == typevar_constructed_id) {
      out_param_types.push_back(ndt::typevar_constructed_type::make(
          t.extended<ndt::typevar_constructed_type>()->get_name(),
          ndt::make_ellipsis_dim(dimsname, t.extended<ndt::typevar_constructed_type>()->get_arg())));
    }
    else {
      out_param_types.push_back(ndt::make_ellipsis_dim(dimsname, t));
    }
  }

  ndt::type kwd_tp = child_tp->get_kwd_struct();
  /*
    if (true) {
      intptr_t old_field_count =
          kwd_tp.extended<base_struct_type>()->get_field_count();
      nd::array names =
          nd::empty(ndt::make_fixed_dim(old_field_count + 2,
    ndt::make_string()));
      nd::array fields =
          nd::empty(ndt::make_fixed_dim(old_field_count + 2, ndt::make_type()));
      for (intptr_t i = 0; i < old_field_count; ++i) {
        names(i)
            .val_assign(kwd_tp.extended<base_struct_type>()->get_field_name(i));
        fields(i)
            .val_assign(kwd_tp.extended<base_struct_type>()->get_field_type(i));
      }
      names(old_field_count).val_assign("threads");
      fields(old_field_count)
          .val_assign(ndt::make_option(ndt::make_type<int>()));
      names(old_field_count + 1).val_assign("blocks");
      fields(old_field_count + 1)
          .val_assign(ndt::make_option(ndt::make_type<int>()));
      kwd_tp = ndt::struct_type::make(names, fields);
    }
  */

  ndt::type ret_tp = child_tp->get_return_type();
  if (ret_tp.get_id() == typevar_constructed_id) {
    ret_tp = ndt::typevar_constructed_type::make(
        ret_tp.extended<ndt::typevar_constructed_type>()->get_name(),
        ndt::make_ellipsis_dim(dimsname, ret_tp.extended<ndt::typevar_constructed_type>()->get_arg()));
  }
  else {
    ret_tp = ndt::make_ellipsis_dim(dimsname, ret_tp);
  }

  return ndt::callable_type::make(ret_tp, ndt::tuple_type::make(out_param_types), kwd_tp);
}

nd::callable nd::functional::elwise(const ndt::type &self_tp, const callable &child)
{
  return callable::make<elwise_virtual_ck>(self_tp, child);
}

nd::callable nd::functional::elwise(const callable &child) { return elwise(elwise_make_type(child.get_type()), child); }
