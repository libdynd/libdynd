//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/functional.hpp>
#include <dynd/index.hpp>
#include <dynd/callables/index_callable.hpp>
#include <dynd/callables/index_dispatch_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::index = nd::make_callable<nd::index_dispatch_callable>(
    ndt::type("(Any, i: Any) -> Any"),
    nd::callable::new_make_all<nd::index_callable, type_id_sequence<int32_id, fixed_dim_id>>());
