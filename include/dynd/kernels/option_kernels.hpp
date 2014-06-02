//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__OPTION_KERNELS_HPP_
#define _DYND__OPTION_KERNELS_HPP_

#include <dynd/func/arrfunc.hpp>

namespace dynd { namespace kernels {

/**
 * Returns the nafunc structure for the given builtin type id.
 */
const nd::array &get_option_builtin_nafunc(type_id_t tid);

}} // namespace dynd::kernels

#endif // _DYND__OPTION_KERNELS_HPP_
