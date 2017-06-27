//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/all_equal_callable.hpp>
#include <dynd/callables/multidispatch_callable.hpp>
#include <dynd/comparison.hpp>
#include <dynd/comparison_common.hpp>
#include <dynd/functional.hpp>

using namespace std;
using namespace dynd;

namespace {
nd::callable make_all_equal() {
  dispatcher<2, nd::callable> dispatcher =
      nd::callable::make_all<nd::all_equal_callable, numeric_types, numeric_types>(func_ptr);

  return nd::functional::reduction(
      [] { return true; },
      nd::make_callable<nd::multidispatch_callable<2>>(
          ndt::make_type<ndt::callable_type>(ndt::make_type<bool>(), {ndt::make_type<ndt::scalar_kind_type>(),
                                                                      ndt::make_type<ndt::scalar_kind_type>()}),
          dispatcher));
}
}

DYND_API nd::callable nd::all_equal = make_all_equal();
