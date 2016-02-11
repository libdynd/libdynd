//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/functional.hpp>
#include <dynd/struct.hpp>
#include <dynd/kernels/field_access_kernel.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::field_access::make()
{
  return callable::make<field_access_kernel>();
}

DYND_API struct nd::field_access nd::field_access;
