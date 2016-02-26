//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  extern DYND_API struct DYND_API all : declfunc<all> {
    static callable make();
    static callable &get();
  } all;

} // namespace dynd::nd
} // namespace dynd
