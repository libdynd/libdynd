//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  extern DYND_API struct DYND_API string_concatenation : declfunc<string_concatenation> {
    static callable make();
  } string_concatenation;

} // namespace dynd::nd
} // namespace dynd
