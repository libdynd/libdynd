//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__PP_ARITHMETIC_HPP_
#define _DYND__PP_ARITHMETIC_HPP_

#include <dynd/pp/gen.hpp>
#include <dynd/pp/list.hpp>

/**
 * Addition of A and B. A, B, and A + B have to be integers between 0 and DYND_PP_LEN_MAX inclusively.
 */
#define DYND_PP_ADD(A, B) DYND_PP_LEN(DYND_PP_MERGE(DYND_PP_ZEROS(A), DYND_PP_ZEROS(B)))

/**
 * Subtraction of B from A. A, B, and B - A have to be integers between 0 and DYND_PP_LEN_MAX inclusively.
 */
#define DYND_PP_SUB(A, B) DYND_PP_LEN(DYND_PP_SLICE_FROM(B, DYND_PP_ZEROS(A)))

#endif
