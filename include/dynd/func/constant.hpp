//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Makes a ckernel that ignores the src values, and writes
     * constant values to the output.
     */
    DYND_API callable constant(const array &val);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
