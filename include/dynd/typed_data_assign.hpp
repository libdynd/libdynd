//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <utility>
#include <iostream>
#include <dynd/visibility.hpp>

namespace dynd {

namespace ndt {
  class DYND_API type;
}

/**
 * Assign one element where src and dst may have different types.
 * Requires that the data be aligned. To assign unaligned data,
 * use ndt::make_unaligned().
 */
DYND_API void typed_data_assign(const ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data,
                                const ndt::type &src_tp, const char *src_arrmeta, const char *src_data,
                                assign_error_mode error_mode = assign_error_fractional);

} // namespace dynd
