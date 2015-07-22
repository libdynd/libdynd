//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/strided_vals.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Create an callable which applies a given window_op in a
     * rolling window fashion.
     *
     * \param neighborhood_op  An callable object which transforms a
     *neighborhood
     *into
     *                         a single output value. Signature
     *                         '(Fixed * Fixed * NH, Fixed * Fixed * MSK) ->
     *OUT',
     */
    callable neighborhood(const callable &neighborhood_op, intptr_t nh_ndim);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
