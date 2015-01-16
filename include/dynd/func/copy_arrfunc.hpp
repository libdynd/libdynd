//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>

namespace dynd {

/**
 * Returns an arrfunc which copies data from one
 * array to another, without broadcasting
 */
const nd::arrfunc& make_copy_arrfunc();

/**
 * Returns an arrfunc which copies data from one
 * array to another, with broadcasting.
 */
const nd::arrfunc& make_broadcast_copy_arrfunc();

} // namespace dynd
