//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/datetime_type.hpp>

namespace dynd {

/**
 * Makes callables which adapt to/from a datetime.
 */
DYND_API bool make_datetime_adapter_callable(const ndt::type &value_tp, const ndt::type &operand_tp,
                                             const std::string &op, nd::callable &out_forward,
                                             nd::callable &out_reverse);

} // namespace dynd
