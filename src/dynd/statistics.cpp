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

namespace {

static std::vector<ndt::type> func_ptr(const ndt::type &DYND_UNUSED(dst_tp), size_t DYND_UNUSED(nsrc),
                                       const ndt::type *src_tp) {
  return {src_tp[0]};
}

} // unnnamed namespace

DYND_API nd::callable nd::max = nd::functional::reduction(nd::make_callable<nd::max_dispatch_callable>(
    ndt::make_type<ndt::callable_type>(ndt::make_type<ndt::scalar_kind_type>(),
                                       {ndt::make_type<ndt::scalar_kind_type>()}),
    nd::callable::make_all<nd::max_callable, arithmetic_types>(func_ptr)));

DYND_API nd::callable nd::mean = nd::make_callable<nd::mean_callable>(ndt::make_type<int64_t>());

DYND_API nd::callable nd::min = nd::functional::reduction(nd::make_callable<nd::min_dispatch_callable>(
    ndt::make_type<ndt::callable_type>(ndt::make_type<ndt::scalar_kind_type>(),
                                       {ndt::make_type<ndt::scalar_kind_type>()}),
    nd::callable::make_all<nd::min_callable, arithmetic_types>(func_ptr)));
