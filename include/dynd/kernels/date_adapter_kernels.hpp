//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef DYND_KERNELS_DATE_ADAPTER_KERNELS_HPP
#define DYND_KERNELS_DATE_ADAPTER_KERNELS_HPP

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/date_type.hpp>

namespace dynd {

/**
 * Makes arrfuncs which adapt to/from a date.
 */
bool make_date_adapter_arrfunc(const ndt::type &operand_tp,
                                 const nd::string &op, nd::arrfunc &out_forward,
                                 nd::arrfunc &out_reverse);

} // namespace dynd

#endif // DYND_KERNELS_DATE_ADAPTER_KERNELS_HPP

