//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/reduction.hpp>
#include <dynd/io.hpp>
#include <dynd/callables/serialize_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::serialize::make()
{
  return functional::reduction(make_callable<serialize_callable<scalar_kind_id>>());
}

DYND_DEFAULT_DECLFUNC_GET(nd::serialize)

DYND_API struct nd::serialize nd::serialize;
