//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/types/type_alignment.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/types/adapt_type.hpp>
#include <dynd/types/fixed_bytes_type.hpp>

using namespace std;
using namespace dynd;

ndt::type ndt::make_unaligned(const ndt::type &value_type)
{
  if (value_type.get_data_alignment() > 1) {
    // Only do something if it requires alignment
    if (value_type.get_kind() != expr_kind) {
      //      return make_type<adapt_type>(value_type, ndt::make_fixed_bytes(value_type.get_data_size(), 1),
      //      nd::callable(),
      //                                 nd::callable());
      return ndt::view_type::make(value_type, ndt::make_fixed_bytes(value_type.get_data_size(), 1));
    }
    else {
      const ndt::type &sdt = value_type.storage_type();
      return ndt::type(value_type.extended<base_expr_type>()->with_replaced_storage_type(
          ndt::view_type::make(sdt, ndt::make_fixed_bytes(sdt.get_data_size(), 1))));
    }
  }
  else {
    return value_type;
  }
}
