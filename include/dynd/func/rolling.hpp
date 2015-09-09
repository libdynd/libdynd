//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/array.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Create a callable which applies a given window_op in a
     * rolling window fashion.
     *
     * \param window_op  A callable object which should be applied to each
     *                   window. The types of this ckernel must match
     *                   appropriately with `dst_tp` and `src_tp`.
     * \param window_size  The size of the rolling window.
     */
    DYND_API callable rolling(const callable &window_op, intptr_t window_size);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
