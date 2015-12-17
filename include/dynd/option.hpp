//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  extern DYND_API struct assign_na : declfunc<assign_na> {
    static DYND_API callable make();
  } assign_na;

  extern DYND_API struct is_avail : declfunc<is_avail> {
    static DYND_API callable make();
  } is_avail;

} // namespace dynd::nd
} // namespace dynd
