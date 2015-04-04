//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/arrmeta_holder.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/outer.hpp>

using namespace std;
using namespace dynd;

nd::arrfunc nd::functional::outer(const arrfunc &child)
{
  return as_arrfunc<outer_ck>(outer_make_type(child.get_type()), child, 0);
}

ndt::type nd::functional::outer_make_type(const arrfunc_type *child_tp)
{
  const ndt::type *param_types = child_tp->get_pos_types_raw();
  intptr_t param_count = child_tp->get_npos();
  dynd::nd::array out_param_types =
      dynd::nd::empty(param_count, ndt::make_type());
  ndt::type *pt =
      reinterpret_cast<ndt::type *>(out_param_types.get_readwrite_originptr());

  for (intptr_t i = 0, i_end = child_tp->get_npos(); i != i_end; ++i) {
    nd::string dimsname("Dims" + std::to_string(i));
    if (param_types[i].get_kind() == memory_kind) {
      pt[i] = pt[i].extended<base_memory_type>()->with_replaced_storage_type(
          ndt::make_ellipsis_dim(dimsname,
                                 param_types[i].without_memory_type()));
    } else if (param_types[i].get_type_id() == typevar_constructed_type_id) {
      pt[i] = ndt::make_typevar_constructed(
          param_types[i].extended<typevar_constructed_type>()->get_name(),
          ndt::make_ellipsis_dim(
              dimsname,
              param_types[i].extended<typevar_constructed_type>()->get_arg()));
    } else {
      pt[i] = ndt::make_ellipsis_dim(dimsname, param_types[i]);
    }
  }

  ndt::type kwd_tp = child_tp->get_kwd_struct();

  ndt::type ret_tp = child_tp->get_return_type();
  if (ret_tp.get_kind() == memory_kind) {
    throw std::runtime_error("outer -- need to fix this");
  } else if (ret_tp.get_type_id() == typevar_constructed_type_id) {
    ret_tp = ndt::make_typevar_constructed(
        ret_tp.extended<typevar_constructed_type>()->get_name(),
        ndt::make_ellipsis_dim(
            "Dims", ret_tp.extended<typevar_constructed_type>()->get_arg()));
  } else {
    ret_tp = ndt::make_ellipsis_dim("Dims", child_tp->get_return_type());
  }

  return ndt::make_arrfunc(ndt::make_tuple(out_param_types), kwd_tp, ret_tp);
}