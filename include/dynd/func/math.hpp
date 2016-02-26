//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/math.hpp>

namespace dynd {
namespace nd {

  extern DYND_API struct DYND_API cos : declfunc<cos> {
    static callable make();
    static callable &get();
  } cos;

  extern DYND_API struct DYND_API sin : declfunc<sin> {
    static callable make();
    static callable &get();
  } sin;

  extern DYND_API struct DYND_API tan : declfunc<tan> {
    static callable make();
    static callable &get();
  } tan;

  extern DYND_API struct DYND_API exp : declfunc<exp> {
    static callable make();
    static callable &get();
  } exp;

} // namespace dynd::nd
} // namespace dynd
