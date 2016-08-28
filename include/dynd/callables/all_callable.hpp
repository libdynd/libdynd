//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/all_kernel.hpp>

namespace dynd {
namespace nd {

  class all_callable : public default_instantiable_callable<all_kernel> {
  public:
    all_callable()
        : default_instantiable_callable<all_kernel>(
              ndt::make_type<ndt::callable_type>(ndt::make_type<bool>(), {ndt::make_type<bool>()})) {}
  };

} // namespace dynd::nd
} // namespace dynd
