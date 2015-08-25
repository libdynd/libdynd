//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/constant.hpp>
#include <dynd/kernels/constant_kernel.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::functional::constant(const array &val)
{
  return callable::make<constant_kernel>(
      ndt::callable_type::make(val.get_type(), ndt::tuple_type::make(true),
                               ndt::struct_type::make(true)),
      val);
}
