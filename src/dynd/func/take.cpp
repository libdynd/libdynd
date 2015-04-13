//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/take.hpp>
#include <dynd/kernels/take.hpp>

using namespace std;
using namespace dynd;

nd::arrfunc nd::take::make()
{
  // Masked take: (M * T, M * bool) -> var * T
  // Indexed take: (M * T, N * intptr) -> N * T
  // Combined: (M * T, N * Ix) -> R * T
  return arrfunc::make<take_ck>(ndt::type("(Dims... * T, N * Ix) -> R * T"), 0);
}

struct nd::take nd::take;