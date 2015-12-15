//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    DYND_API callable convert(const ndt::type &tp, const callable &child);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
