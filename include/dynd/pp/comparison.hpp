//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__PP_COMPARISON_HPP_
#define _DYND__PP_COMPARISON_HPP_

#include <dynd/pp/gen.hpp>
#include <dynd/pp/list.hpp>
#include <dynd/pp/logical.hpp>
#include <dynd/pp/token.hpp>

/**
 * Less than comparison between two tokens. A and B have to be integers
 * between 0 and DYND_PP_LEN_MAX inclusively.
 */
#define DYND_PP_LT(A, B) DYND_PP_PASTE(DYND_PP_LT_, DYND_PP_BOOL(B))(A, B)
#define DYND_PP_LT_0 DYND_PP_TO_ZERO
#define DYND_PP_LT_1(A, B) DYND_PP_GET(A, DYND_PP_MERGE(DYND_PP_ONES(B), DYND_PP_ZEROS(A)))

/**
 * Less equal comparison between two tokens. A and B have to be integers
 * between 0 and DYND_PP_LEN_MAX inclusively.
 */
#define DYND_PP_LE(A, B) DYND_PP_GE(B, A)

/**
 * Equality comparison between two tokens. A and B do not have to be integers.
 * Expands to 1 if either DYND_PP_A_EQ_B or DYND_PP_B_EQ_A is defined as a macro. Otherwise 0.
 */
#define DYND_PP_EQ(A, B) DYND_PP_PASTE(DYND_PP_EQ_, \
    DYND_PP_PASTE(DYND_PP_IS_NULL(DYND_PP_PASTE(DYND_PP_, DYND_PP_PASTE(A, DYND_PP_PASTE(_EQ_, B)))), \
    DYND_PP_PASTE(_, DYND_PP_IS_NULL(DYND_PP_PASTE(DYND_PP_, DYND_PP_PASTE(B, DYND_PP_PASTE(_EQ_, A)))))))
#define DYND_PP_EQ_0_0 0
#define DYND_PP_EQ_0_1 1
#define DYND_PP_EQ_1_0 1
#define DYND_PP_EQ_1_1 1

/**
 * Inequality comparison between two tokens. A and B do not have to be integers.
 * Expands to 0 if either DYND_PP_A_EQ_B or DYND_PP_B_EQ_A is defined as a macro. Otherwise 1.
 */
#define DYND_PP_NE(A, B) DYND_PP_NOT(DYND_PP_EQ(A, B))

/**
 * Greater equal comparison between two tokens. A and B have to be integers
 * between 0 and DYND_PP_LEN_MAX inclusively.
 */
#define DYND_PP_GE(A, B) DYND_PP_PASTE(DYND_PP_GE_, DYND_PP_BOOL(B))(A, B)
#define DYND_PP_GE_0 DYND_PP_TO_ONE
#define DYND_PP_GE_1(A, B) DYND_PP_GET(A, DYND_PP_MERGE(DYND_PP_ZEROS(B), DYND_PP_ONES(A)))

/**
 * Greater than comparison between two tokens. A and B have to be integers
 * between 0 and DYND_PP_LEN_MAX inclusively.
 */
#define DYND_PP_GT(A, B) DYND_PP_LT(B, A)

#endif
