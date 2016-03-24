//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/convert.hpp>
#include <dynd/callables/convert_callable.hpp>

using namespace std;
using namespace dynd;

nd::callable nd::functional::convert(const ndt::type &tp, const callable &child)
{
  return make_callable<convert_callable>(tp, child);
}
