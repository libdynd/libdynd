//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_instantiable_callable.hpp>
#include <dynd/kernels/string_count_kernel.hpp>

namespace dynd {
namespace nd {

  class string_count_callable : public base_instantiable_callable<string_count_kernel> {
  public:
    string_count_callable()
        : base_instantiable_callable<string_count_kernel>(
              ndt::callable_type::make(ndt::make_type<intptr_t>(), {ndt::type(string_id), ndt::type(string_id)}))
    {
    }
  };

} // namespace dynd::nd
} // namespace dynd
