//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/callable.hpp>

namespace dynd {
namespace nd {

  extern struct sum : declfunc<sum> {
    static callable make();
  } sum;

} // namespace dynd::nd
} // namespace dynd
