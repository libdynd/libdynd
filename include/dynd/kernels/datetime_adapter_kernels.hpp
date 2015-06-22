//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/datetime_type.hpp>

namespace dynd {

/**
 * Makes arrfuncs which adapt to/from a datetime.
 */
bool make_datetime_adapter_arrfunc(const ndt::type &value_tp,
                                   const ndt::type &operand_tp,
                                   const nd::string &op,
                                   nd::arrfunc &out_forward,
                                   nd::arrfunc &out_reverse);

} // namespace dynd
