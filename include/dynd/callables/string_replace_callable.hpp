//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/string_replace_kernel.hpp>

namespace dynd {
namespace nd {

  class string_replace_callable : public default_instantiable_callable<string_replace_kernel> {
  public:
    string_replace_callable()
        : default_instantiable_callable<string_replace_kernel>(ndt::callable_type::make(
              ndt::type(string_id), {ndt::type(string_id), ndt::type(string_id), ndt::type(string_id)}))
    {
    }
  };

} // namespace dynd::nd
} // namespace dynd
