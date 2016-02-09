//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  extern DYND_API struct DYND_API real : declfunc<real> {
    static callable make();
  } real;

} // namespace dynd::nd
} // namespace dynd
