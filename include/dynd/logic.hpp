//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  extern DYND_API struct all : declfunc<all> {
    static DYND_API callable make();
  } all;

} // namespace dynd::nd
} // namespace dynd
