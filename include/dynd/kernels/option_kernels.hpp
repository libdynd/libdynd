//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>
#include <dynd/types/option_type.hpp>

namespace dynd {

/**
 * Returns the nafunc structure for the given builtin type id.
 */
const nd::array &get_option_builtin_nafunc(type_id_t tid);

} // namespace dynd