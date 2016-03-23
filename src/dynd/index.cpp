//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/functional.hpp>
#include <dynd/index.hpp>
#include <dynd/kernels/index_kernel.hpp>
#include <dynd/callables/index_dispatch_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::index::make()
{
  typedef type_id_sequence<int32_id, fixed_dim_id> type_ids;

  auto dispatcher = callable::new_make_all<index_kernel, type_ids>();
  return make_callable<index_dispatch_callable>(ndt::type("(Any, i: Any) -> Any"), dispatcher);
}

DYND_DEFAULT_DECLFUNC_GET(nd::index)

DYND_API struct nd::index nd::index;
