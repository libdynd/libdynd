//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/strided_vals.hpp>
#include <dynd/func/callable.hpp>

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
    DYND_API callable neighborhood(const callable &child, const callable &boundary_child = callable());

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
