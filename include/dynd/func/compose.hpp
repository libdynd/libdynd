//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Returns an callable which composes the two callables together.
     * The buffer used to connect them is made out of the provided ``buf_tp``.
     */
    DYND_API callable compose(const callable &first, const callable &second,
                              const ndt::type &buf_tp = ndt::type());

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
