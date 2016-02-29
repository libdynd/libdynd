//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  /**
   * Performs a binary search of the first dimension of the array, which
   * should be sorted.
   *
   * \returns  The index of the found element, or -1 if not found.
   */
  extern DYND_API struct DYND_API binary_search : declfunc<binary_search> {
    static callable make();
    static callable &get();
  } binary_search;

} // namespace dynd::nd
} // namespace dynd
