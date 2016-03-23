//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/max.hpp>
#include <dynd/func/reduction.hpp>
#include <dynd/functional.hpp>
#include <dynd/kernels/max_kernel.hpp>
#include <dynd/types/scalar_kind_type.hpp>
#include <dynd/callables/max_dispatch_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::max::make()
{
  auto dispatcher = callable::new_make_all<max_kernel, arithmetic_ids>();

  return functional::reduction(make_callable<max_dispatch_callable>(
      ndt::callable_type::make(ndt::scalar_kind_type::make(), ndt::scalar_kind_type::make()), dispatcher));
}

DYND_DEFAULT_DECLFUNC_GET(nd::max)

DYND_API struct nd::max nd::max;
