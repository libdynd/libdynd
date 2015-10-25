//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/sort.hpp>
#include <dynd/kernels/sort_kernel.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::sort::make()
{
  return callable::make<sort_kernel>();
}

DYND_API struct nd::sort nd::sort;
