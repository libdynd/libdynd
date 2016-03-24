//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/min.hpp>
#include <dynd/func/reduction.hpp>
#include <dynd/functional.hpp>
#include <dynd/callables/min_callable.hpp>
#include <dynd/types/scalar_kind_type.hpp>
#include <dynd/callables/min_dispatch_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::min::make()
{
  auto dispatcher = callable::new_make_all<min_callable, arithmetic_ids>();

  return functional::reduction(make_callable<min_dispatch_callable>(
      ndt::callable_type::make(ndt::scalar_kind_type::make(), ndt::scalar_kind_type::make()), dispatcher));
}

DYND_DEFAULT_DECLFUNC_GET(nd::min)

DYND_API struct nd::min nd::min;
