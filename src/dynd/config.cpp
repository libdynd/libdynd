//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/config.hpp>

using namespace std;
using namespace dynd;

bool dynd::built_with_cuda()
{
#ifdef DYND_CUDA
  return true;
#else
  return false;
#endif
}

std::ostream &dynd::operator<<(ostream &o, assign_error_mode errmode)
{
  switch (errmode) {
  case assign_error_nocheck:
    o << "nocheck";
    break;
  case assign_error_overflow:
    o << "overflow";
    break;
  case assign_error_fractional:
    o << "fractional";
    break;
  case assign_error_inexact:
    o << "inexact";
    break;
  case assign_error_default:
    o << "default";
    break;
  default:
    o << "invalid error mode(" << (int)errmode << ")";
    break;
  }

  return o;
}
