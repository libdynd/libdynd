//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>

namespace dynd {
namespace nd {

  class base_callable;

  /**
   * Aligns a size as required by kernels.
   */
  static constexpr size_t aligned_size(size_t size)
  {
    return (size + static_cast<size_t>(7)) & ~static_cast<size_t>(7);
  }

} // namespace dynd::nd
} // namespace dynd
