//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/callables/not_equal_callable.hpp>
#include <dynd/comparison_common.hpp>

using namespace std;
using namespace dynd;

namespace {
nd::callable make_not_equal() {
  dispatcher<2, nd::callable> dispatcher = make_comparison_children<func_ptr, nd::not_equal_callable>();
  dispatcher.insert(nd::make_callable<nd::not_equal_callable<dynd::complex<float>, dynd::complex<float>>>());
  dispatcher.insert(nd::make_callable<nd::not_equal_callable<dynd::complex<double>, dynd::complex<double>>>());
  dispatcher.insert(nd::make_callable<nd::not_equal_callable<ndt::tuple_type, ndt::tuple_type>>());
  dispatcher.insert(nd::make_callable<nd::not_equal_callable<ndt::struct_type, ndt::struct_type>>());
  dispatcher.insert(nd::make_callable<nd::not_equal_callable<ndt::type, ndt::type>>());
  dispatcher.insert(nd::make_callable<nd::not_equal_callable<bytes, bytes>>());

  return nd::make_callable<nd::multidispatch_callable<2>>(
      ndt::make_type<ndt::callable_type>(ndt::make_type<ndt::any_kind_type>(),
                                         {ndt::make_type<ndt::any_kind_type>(), ndt::make_type<ndt::any_kind_type>()}),
      dispatcher);
}
}

DYND_API nd::callable nd::not_equal = make_not_equal();
