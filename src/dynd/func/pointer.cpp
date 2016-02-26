//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/pointer.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/dereference_kernel.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::dereference::make()
{
  return nd::callable::make<dereference_kernel>(ndt::type("(self: Any) -> Any"));
}

DYND_DEFAULT_DECLFUNC_GET(nd::dereference)

DYND_API struct nd::dereference nd::dereference;
