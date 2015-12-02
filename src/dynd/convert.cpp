//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/convert.hpp>
#include <dynd/type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/convert_kernel.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::functional::convert(ndt::type &&, callable &&)
{
  throw std::runtime_error("");
//  return callable::make<convert_kernel>(std::forward<ndt::type>(tp), std::forward<callable>(child));
}
