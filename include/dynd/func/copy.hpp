//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  /**
   * Returns an arrfunc which copies data from one
   * array to another, without broadcasting
   */
  extern DYND_API callable copy;

} // namespace dynd::nd
} // namespace dynd
