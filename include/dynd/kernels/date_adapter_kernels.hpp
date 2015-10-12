//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/assignment.hpp>
#include <dynd/types/date_type.hpp>

namespace dynd {

/**
 * Makes callables which adapt to/from a date.
 */
bool DYND_API make_date_adapter_callable(const ndt::type &operand_tp, const std::string &op, nd::callable &out_forward,
                                         nd::callable &out_reverse);

} // namespace dynd
