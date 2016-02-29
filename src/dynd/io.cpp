//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/reduction.hpp>
#include <dynd/io.hpp>
#include <dynd/kernels/serialize_kernel.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::serialize::make()
{
  return functional::reduction(callable::make<serialize_kernel<scalar_kind_id>>());
}

DYND_DEFAULT_DECLFUNC_GET(nd::serialize)

DYND_API struct nd::serialize nd::serialize;
