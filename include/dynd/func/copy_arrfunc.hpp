//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND_FUNC_COPY_ARRFUNC_HPP
#define DYND_FUNC_COPY_ARRFUNC_HPP

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

#endif // DYND_FUNC_COPY_ARRFUNC_HPP
