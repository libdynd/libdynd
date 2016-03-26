//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/func/max.hpp>
#include <dynd/func/reduction.hpp>
#include <dynd/functional.hpp>
#include <dynd/callables/max_callable.hpp>
#include <dynd/types/scalar_kind_type.hpp>
#include <dynd/callables/max_dispatch_callable.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::max = nd::functional::reduction(nd::make_callable<nd::max_dispatch_callable>(
    ndt::callable_type::make(ndt::scalar_kind_type::make(), ndt::scalar_kind_type::make()),
    nd::callable::new_make_all<nd::max_callable, arithmetic_ids>()));
