//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/index_callable.hpp>
#include <dynd/callables/index_dispatch_callable.hpp>
#include <dynd/callables/take_dispatch_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/index.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::index = nd::make_callable<nd::index_dispatch_callable>(
    ndt::type("(Any, i: Any) -> Any"),
    nd::callable::make_all<nd::index_callable, type_sequence<int32_t, ndt::fixed_dim_kind_type>>());

DYND_API nd::callable nd::take = nd::make_callable<nd::take_dispatch_callable>();
