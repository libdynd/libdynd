//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/callable.hpp>
#include <dynd/kernels/outer.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    DYND_API callable outer(const callable &child);

    DYND_API ndt::type outer_make_type(const ndt::callable_type *child_tp);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
