//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/type_id.hpp>
#include <dynd/kernels/ckernel_prefix.hpp>
#include <dynd/math.hpp>

namespace dynd {

enum comparison_type_t {
  /**
   * A less than operation suitable for sorting
   * (one of a < b or b < a must be true when a != b).
   */
  comparison_type_sorting_less,
  /** Standard comparisons */
  comparison_type_less,
  comparison_type_less_equal,
  comparison_type_equal,
  comparison_type_not_equal,
  comparison_type_greater_equal,
  comparison_type_greater
};

} // namespace dynd
