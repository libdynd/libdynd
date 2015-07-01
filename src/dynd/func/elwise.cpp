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

ndt::type nd::functional::elwise_make_type(const ndt::arrfunc_type *child_tp)
{
  const ndt::type *param_types = child_tp->get_pos_types_raw();
  intptr_t param_count = child_tp->get_npos();
  dynd::nd::array out_param_types =
      dynd::nd::empty(param_count, ndt::make_type());
  dynd::nd::string dimsname("Dims");
  ndt::type *pt =
      reinterpret_cast<ndt::type *>(out_param_types.get_readwrite_originptr());

  for (intptr_t i = 0, i_end = child_tp->get_npos(); i != i_end; ++i) {
    if (param_types[i].get_kind() == memory_kind) {
      pt[i] =
          pt[i].extended<ndt::base_memory_type>()->with_replaced_storage_type(
              ndt::make_ellipsis_dim(dimsname,
                                     param_types[i].without_memory_type()));
    } else if (param_types[i].get_type_id() == typevar_constructed_type_id) {
      pt[i] = ndt::make_typevar_constructed(
          param_types[i].extended<ndt::typevar_constructed_type>()->get_name(),
          ndt::make_ellipsis_dim(dimsname,
                                 param_types[i]
                                     .extended<ndt::typevar_constructed_type>()
                                     ->get_arg()));
    } else {
      pt[i] = ndt::make_ellipsis_dim(dimsname, param_types[i]);
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
      kwd_tp = ndt::make_struct(names, fields);
    }
  */

  ndt::type ret_tp = child_tp->get_return_type();
  if (ret_tp.get_kind() == memory_kind) {
    ret_tp =
        ret_tp.extended<ndt::base_memory_type>()->with_replaced_storage_type(
            ndt::make_ellipsis_dim(dimsname, ret_tp.without_memory_type()));
  } else if (ret_tp.get_type_id() == typevar_constructed_type_id) {
    ret_tp = ndt::make_typevar_constructed(
        ret_tp.extended<ndt::typevar_constructed_type>()->get_name(),
        ndt::make_ellipsis_dim(
            dimsname,
            ret_tp.extended<ndt::typevar_constructed_type>()->get_arg()));
  } else {
    ret_tp = ndt::make_ellipsis_dim(dimsname, ret_tp);
  }

  return ndt::make_arrfunc(ndt::make_tuple(out_param_types), kwd_tp, ret_tp);
}

nd::arrfunc nd::functional::elwise(const ndt::type &self_tp,
                                   const arrfunc &child)
{
  if (child.get()->data_init == NULL) {
    throw std::runtime_error("elwise child has NULL data_init");
  }
  if (child.get()->resolve_dst_type == NULL) {
    throw std::runtime_error("elwise child has NULL resolve_dst_type");
  }

  return arrfunc::make<elwise_virtual_ck>(
      self_tp, child, child.get()->data_size + sizeof(ndt::type));
}

nd::arrfunc nd::functional::elwise(const arrfunc &child)
{
  return elwise(elwise_make_type(child.get_type()), child);
}