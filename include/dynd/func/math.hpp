//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/math.hpp>

namespace dynd {
namespace nd {

  extern DYND_API struct cos : declfunc<cos> {
    static DYND_API callable make();
  } cos;

  extern DYND_API struct sin : declfunc<sin> {
    static DYND_API callable make();
  } sin;

  extern DYND_API struct tan : declfunc<tan> {
    static DYND_API callable make();
  } tan;

  extern DYND_API struct exp : declfunc<exp> {
    static DYND_API callable make();
  } exp;

} // namespace dynd::nd
} // namespace dynd
