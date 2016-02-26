//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {

  extern DYND_API struct DYND_API real : declfunc<real> {
    static callable make();
    static callable &get();
  } real;

  extern DYND_API struct DYND_API imag : declfunc<imag> {
    static callable make();
    static callable &get();
  } imag;

  extern DYND_API struct DYND_API conj : declfunc<conj> {
    static callable make();
    static callable &get();
  } conj;

} // namespace dynd::nd
} // namespace dynd
