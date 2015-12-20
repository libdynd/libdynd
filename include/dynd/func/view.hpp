//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  extern DYND_API struct view : declfunc<view> {
    static DYND_API callable make();
  } view;

} // namespace dynd::nd
} // namespace dynd
