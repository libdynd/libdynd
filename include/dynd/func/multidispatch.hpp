//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>

namespace dynd {
namespace nd {
  namespace functional {

    /**
     * Creates a multiple dispatch arrfunc out of a set of arrfuncs. The
     * input arrfuncs must have concrete signatures.
     *
     * \param naf  The number of arrfuncs provided.
     * \param af  The array of input arrfuncs, sized ``naf``.
     */
    arrfunc multidispatch(intptr_t naf, const arrfunc *af);

    template <typename... A>
    arrfunc multidispatch(arrfunc a0, A &&... a)
    {
      arrfunc af[1 + sizeof...(A)] = {a0, a...};
      return multidispatch(1 + sizeof...(A), af);
    }

  } // namespace dynd::nd::functional
} // namespace dynd::nd
} // namespace dynd
