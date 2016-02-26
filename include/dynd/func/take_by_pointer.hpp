//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/pointer_type.hpp>

namespace dynd {
namespace nd {

  /**
   * Create an callable which applies an indexed take/"fancy indexing"
   * operation, but stores the pointers.
   *
   */
  extern DYND_API struct DYND_API take_by_pointer : declfunc<take_by_pointer> {
    static callable make();
    static callable &get();
  } take_by_pointer;

} // namespace dynd::nd
} // namespace dynd
