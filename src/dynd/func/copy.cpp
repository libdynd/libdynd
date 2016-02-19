//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/copy.hpp>
#include <dynd/kernels/copy_kernel.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::copy::make()
{
  return callable::make<copy_ck>(ndt::type("(A... * S) -> B... * T"), 0);
}

DYND_API struct nd::copy nd::copy;
