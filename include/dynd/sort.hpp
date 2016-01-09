//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  extern DYND_API struct DYND_API sort : declfunc<sort> {
    static callable make();
  } sort;

  extern DYND_API struct DYND_API unique : declfunc<unique> {
    static callable make();
  } unique;

} // namespace dynd::nd
} // namespace dynd
