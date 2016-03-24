//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callables/base_instantiable_callable.hpp>
#include <dynd/kernels/elwise.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    class elwise_callable : public base_callable {
    public:
      elwise_callable() : base_callable(ndt::type("(bool) -> bool")) {}
    };

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
