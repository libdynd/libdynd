//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/type.hpp>

namespace dynd {

/**
 * Given two types, this function produces the type with which
 * to do arithmetic calculations for both (i.e. float + int -> float)
 *
 * For the built-in types, this is generally following the
 * rules for C/C++, with a unit test validating the results.
 *
 * If the inputs are in NBO (native byte order), the result will
 * be in NBO. If the inputs are not, the output may or may not
 * be in NBO, the function makes no effort to follow a convention.
 */
DYND_API ndt::type promote_types_arithmetic(const ndt::type& tp0, const ndt::type& tp1);

} // namespace dynd
