//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/config.hpp>

using namespace std;
using namespace dynd;

bool built_with_cuda()
{
#ifdef DYND_CUDA
  return true;
#else
  return false;
#endif
}