//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_instantiable_callable.hpp>
#include <dynd/kernels/string_concat_kernel.hpp>

namespace dynd {
namespace nd {

  class string_concat_callable : public base_instantiable_callable<string_concatenation_kernel> {
  public:
    string_concat_callable()
        : base_instantiable_callable<string_concatenation_kernel>(
              ndt::callable_type::make(ndt::type(string_id), {ndt::type(string_id), ndt::type(string_id)}))
    {
    }
  };

} // namespace dynd::nd
} // namespace dynd
