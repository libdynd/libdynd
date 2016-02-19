//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/string_contains_kernel.hpp>

namespace dynd {
namespace nd {

  class string_contains_callable : public default_instantiable_callable<string_contains_kernel> {
  public:
    string_contains_callable()
        : default_instantiable_callable<string_contains_kernel>(
              ndt::make_type<ndt::callable_type>(ndt::type(bool_id), {ndt::type(string_id), ndt::type(string_id)})) {}
  };

} // namespace dynd::nd
} // namespace dynd
