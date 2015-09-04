//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/math.hpp>

namespace dynd {
namespace nd {

  extern struct cos : declfunc<cos> {
    static callable make();
  } cos;

  extern struct sin : declfunc<sin> {
    static callable make();
  } sin;

  extern struct tan : declfunc<tan> {
    static callable make();
  } tan;

  extern struct exp : declfunc<exp> {
    static callable make();
  } exp;

} // namespace dynd::nd
} // namespace dynd
