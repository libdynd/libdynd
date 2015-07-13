//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>
#include <dynd/types/option_type.hpp>

namespace dynd {

/**
 * Returns the builtin is_avail for the given builtin type id.
 */
const nd::arrfunc &get_option_builtin_is_avail(type_id_t tid);

/**
 * Returns the builtin assign_na for the given builtin type id.
 */
const nd::arrfunc &get_option_builtin_assign_na(type_id_t tid);

} // namespace dynd