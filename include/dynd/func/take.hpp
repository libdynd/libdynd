//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  /**
   * An callable which applies either a boolean masked or
   * an indexed take/"fancy indexing" operation.
   */
  extern DYND_API struct DYND_API take : declfunc<take> {
    static callable make();
    static callable &get();
  } take;

} // namespace dynd::nd
} // namespace dynd
