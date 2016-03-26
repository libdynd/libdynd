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

DYND_API nd::callable nd::min = nd::functional::reduction(nd::make_callable<nd::min_dispatch_callable>(
    ndt::callable_type::make(ndt::scalar_kind_type::make(), ndt::scalar_kind_type::make()),
    nd::callable::new_make_all<nd::min_callable, arithmetic_ids>()));
