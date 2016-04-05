//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/storagebuf.hpp>

namespace dynd {
namespace nd {

  struct kernel_prefix;

  /**
   * Function pointers + data for a hierarchical
   * kernel which operates on type/arrmeta in
   * some configuration.
   *
   * The data placed in the kernel's data must
   * be relocatable with a memcpy, it must not rely on its
   * own address.
   */
  class kernel_builder : public storagebuf<kernel_prefix, kernel_builder> {
  public:
    DYND_API void destroy();

    ~kernel_builder() { destroy(); }
  };

} // namespace dynd::nd
} // namespace dynd
