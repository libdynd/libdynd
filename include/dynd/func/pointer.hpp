//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  extern DYND_API struct DYND_API dereference : declfunc<dereference> {
    static callable make();
    static callable &get();
  } dereference;

} // namespace dynd::nd
} // namespace dynd
