//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/view.hpp>
#include <dynd/kernels/view_kernel.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::view::make()
{
  return callable::make<view_kernel>(ndt::type("(Any) -> Any"));
}

DYND_API struct nd::view nd::view;
