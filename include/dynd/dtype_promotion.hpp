//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__DTYPE_PROMOTION_HPP_
#define _DYND__DTYPE_PROMOTION_HPP_

#include <dynd/type.hpp>

namespace dynd {

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
ndt::type promote_dtypes_arithmetic(const ndt::type& dt0, const ndt::type& dt1);

} // namespace dynd

#endif // _DYND__DTYPE_PROMOTION_HPP_
