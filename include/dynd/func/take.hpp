//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/callable.hpp>

namespace dynd {
namespace nd {

  /**
   * An callable which applies either a boolean masked or
   * an indexed take/"fancy indexing" operation.
   */
  extern struct take : declfunc<take> {
    static callable make();
  } take;

} // namespace dynd::nd
} // namespace dynd