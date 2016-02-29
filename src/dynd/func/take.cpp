//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/take.hpp>
#include <dynd/kernels/take_kernel.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::take::make()
{
  // Masked take: (M * T, M * bool) -> var * T
  // Indexed take: (M * T, N * intptr) -> N * T
  // Combined: (M * T, N * Ix) -> R * T
  return callable::make<take_ck>(ndt::type("(Dims... * T, N * Ix) -> R * T"), 0);
}

DYND_DEFAULT_DECLFUNC_GET(nd::take)

DYND_API struct nd::take nd::take;
