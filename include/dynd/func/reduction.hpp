//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Lifts the provided callable, broadcasting it as necessary to execute
     * across the additional dimensions in the ``lifted_types`` array.
     */
    DYND_API callable reduction(const callable &child);

    DYND_API callable reduction(const callable &child,
                                const std::initializer_list<std::pair<const char *, array>> &kwds);

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
