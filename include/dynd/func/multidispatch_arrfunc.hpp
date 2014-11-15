//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/config.hpp>
#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>

namespace dynd {

/**
 * Creates a multiple dispatch arrfunc out of a set of arrfuncs. The
 * input arrfuncs must have concrete signatures.
 *
 * \param naf  The number of arrfuncs provided.
 * \param af  The array of input arrfuncs, sized ``naf``.
 */
nd::arrfunc make_multidispatch_arrfunc(intptr_t naf, const nd::arrfunc *af);

} // namespace dynd
