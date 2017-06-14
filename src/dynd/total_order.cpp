//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/total_order_callable.hpp>
#include <dynd/comparison_common.hpp>

using namespace std;
using namespace dynd;

namespace {
nd::callable make_total_order() {
  return nd::make_callable<nd::multidispatch_callable<2>>(
      ndt::make_type<ndt::callable_type>(ndt::make_type<ndt::any_kind_type>(),
                                         {ndt::make_type<ndt::any_kind_type>(), ndt::make_type<ndt::any_kind_type>()}),
      dispatcher<2, nd::callable>(func_ptr, {nd::make_callable<nd::total_order_callable<dynd::string, dynd::string>>(),
                                             nd::make_callable<nd::total_order_callable<int32_t, int32_t>>(),
                                             nd::make_callable<nd::total_order_callable<bool, bool>>()}));
}
}

DYND_API nd::callable nd::total_order = make_total_order();
