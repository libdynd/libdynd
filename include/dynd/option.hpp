//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  extern DYND_API struct DYND_API assign_na : declfunc<assign_na> {
    static callable make();
    static callable &get();
  } assign_na;

  extern DYND_API struct DYND_API is_na : declfunc<is_na> {
    static callable make();
    static callable &get();
  } is_na;

} // namespace dynd::nd
} // namespace dynd
