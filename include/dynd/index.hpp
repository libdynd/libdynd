//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  extern DYND_API callable index;

  /**
   * An callable which applies either a boolean masked or
   * an indexed take/"fancy indexing" operation.
   */
  extern DYND_API callable take;

} // namespace dynd::nd
} // namespace dynd
