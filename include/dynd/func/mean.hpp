//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/callable.hpp>

namespace dynd {
namespace nd {

  /**
   * Makes a 1D mean callable.
   * (Fixed * <tid>) -> <tid>
   */
  callable make_builtin_mean1d_callable(type_id_t tid, intptr_t minp);

  extern struct mean : declfunc<mean> {
    static callable make();
  } mean;

} // namespace dynd::nd
} // namespace dynd
