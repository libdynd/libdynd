//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/max_callable.hpp>
#include <dynd/callables/max_dispatch_callable.hpp>
#include <dynd/callables/mean_callable.hpp>
#include <dynd/callables/min_callable.hpp>
#include <dynd/callables/min_dispatch_callable.hpp>
#include <dynd/functional.hpp>
#include <dynd/statistics.hpp>
#include <dynd/types/scalar_kind_type.hpp>

using namespace std;
using namespace dynd;

DYND_API nd::callable nd::max = nd::functional::reduction(nd::make_callable<nd::max_dispatch_callable>(
    ndt::make_type<ndt::callable_type>(ndt::make_type<ndt::scalar_kind_type>(),
                                       {ndt::make_type<ndt::scalar_kind_type>()}),
    nd::callable::new_make_all<nd::max_callable, arithmetic_ids>()));

DYND_API nd::callable nd::mean = nd::make_callable<nd::mean_callable>(ndt::type(int64_id));

DYND_API nd::callable nd::min = nd::functional::reduction(nd::make_callable<nd::min_dispatch_callable>(
    ndt::make_type<ndt::callable_type>(ndt::make_type<ndt::scalar_kind_type>(),
                                       {ndt::make_type<ndt::scalar_kind_type>()}),
    nd::callable::new_make_all<nd::min_callable, arithmetic_ids>()));
