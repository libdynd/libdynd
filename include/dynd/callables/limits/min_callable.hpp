//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/default_instantiable_callable.hpp>
#include <dynd/kernels/limits/min_kernel.hpp>

namespace dynd {
namespace nd {
  namespace limits {

    template <typename T>
    class min_callable : public default_instantiable_callable<min_kernel<T>> {
    public:
      min_callable()
          : default_instantiable_callable<min_kernel<T>>(ndt::make_type<ndt::callable_type>(ndt::make_type<T>(), {})) {}
    };

  } // namespace dynd::nd::limits
} // namespace dynd::nd
} // namespace dynd
