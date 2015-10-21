//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/search.hpp>
#include <dynd/kernels/binary_search_kernel.hpp>
#include <dynd/types/fixed_dim_type.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::binary_search::make()
{
  return callable::make<binary_search_kernel>();
}

DYND_API struct nd::binary_search nd::binary_search;
