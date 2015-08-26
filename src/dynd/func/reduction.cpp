//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/compound.hpp>
#include <dynd/func/reduction.hpp>
#include <dynd/kernels/reduction_kernel.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/ellipsis_dim_type.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::functional::reduction(const callable &child)
{
  if (child.is_null()) {
    throw invalid_argument("'child' cannot be null");
  }

  switch (child.get_narg()) {
  case 1:
    break;
  case 2:
    return reduction((child.get_flags() | right_associative) ? left_compound(child) : right_compound(child));
  default: {
    stringstream ss;
    ss << "'child' must be a unary callable, but its signature is " << child.get_array_type();
    throw invalid_argument(ss.str());
  }
  }

  return callable::make<reduction_virtual_kernel>(
      ndt::callable_type::make(ndt::ellipsis_dim_type::make_if_not_variadic(child.get_ret_type()),
                               {ndt::ellipsis_dim_type::make_if_not_variadic(child.get_arg_type(0))},
                               {"axes", "identity", "keepdims"}, {ndt::option_type::make(ndt::type("Fixed * int32")),
                                                                  ndt::option_type::make(child.get_ret_type()),
                                                                  ndt::option_type::make(ndt::type::make<bool1>())}),
      reduction_virtual_kernel::static_data_type(child), sizeof(reduction_virtual_kernel::data_type));
}
