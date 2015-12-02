//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/convert.hpp>
#include <dynd/kernels/convert_kernel.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::functional::convert(const ndt::type &tp, const callable &child)
{
  return callable::make<convert_kernel>(tp, child);
}
