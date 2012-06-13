//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__DTYPE_PROMOTION_HPP_
#define _DND__DTYPE_PROMOTION_HPP_

#include <dnd/dtype.hpp>

namespace dnd {

/**
 * Given two dtypes, this function produces the dtype with which
 * to do arithmetic calculations for both (i.e. float + int -> float)
 *
 * For the built-in types, this is generally following the
 * rules for C/C++, with a unit test validating the results.
 *
 * If the inputs are in NBO (native byte order), the result will
 * be in NBO. If the inputs are not, the output may or may not
 * be in NBO, the function makes no effort to follow a convention.
 */
dtype promote_dtypes_arithmetic(const dtype& dt0, const dtype& dt1);

} // namespace dnd

#endif // _DND__DTYPE_PROMOTION_HPP_
