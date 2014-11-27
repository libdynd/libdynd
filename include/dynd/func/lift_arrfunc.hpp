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
 * Lifts the provided arrfunc, so it broadcasts all the arguments.
 *
 * \param child_af  The arrfunc to be lifted.
 */
nd::arrfunc lift_arrfunc(const nd::arrfunc &child_af);

} // namespace dynd
