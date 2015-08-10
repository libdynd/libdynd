//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/compound_kernel.hpp>
#include <dynd/func/compound.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::functional::left_compound(const callable &child)
{
  return callable::make<left_compound_kernel>(
      ndt::callable_type::make(child.get_type()->get_return_type(),
                               child.get_type()->get_pos_types()(irange() < 1)),
      child, 0);
}

nd::callable nd::functional::right_compound(const callable &child)
{
  return callable::make<right_compound_kernel>(
      ndt::callable_type::make(
          child.get_type()->get_return_type(),
          child.get_type()->get_pos_types()(1 >= irange())),
      child, 0);
}
