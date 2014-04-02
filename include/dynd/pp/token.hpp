//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__PP_TOKEN_HPP_
#define _DYND__PP_TOKEN_HPP_

#include <dynd/pp/gen.hpp>

#define DYND_PP_PASTE(A, B) DYND_PP__PASTE(A, B)
#define DYND_PP__PASTE(A, B) DYND_PP___PASTE(A, B)
#define DYND_PP___PASTE(A, B) A ## B

#define DYND_PP_TO_NULL(...)

#define DYND_PP_TO_ZERO(...) 0
#define DYND_PP_TO_ONE(...) 1

#define DYND_PP_TO_COMMA(...) ,

#define DYND_PP_ID(...) __VA_ARGS__

/*
 * Expands to 1 if __VA_ARGS__ is whitespace. Otherwise 0.
 */
#define DYND_PP_IS_NULL(...) DYND_PP__IS_NULL(DYND_PP_HAS_COMMA(__VA_ARGS__), \
    DYND_PP_HAS_COMMA(DYND_PP_TO_COMMA __VA_ARGS__), DYND_PP_HAS_COMMA(__VA_ARGS__ ()), \
    DYND_PP_HAS_COMMA(DYND_PP_TO_COMMA __VA_ARGS__ ()))
#define DYND_PP__IS_NULL(A, B, C, D) DYND_PP_HAS_COMMA(DYND_PP_PASTE(DYND_PP__IS_NULL_, \
    DYND_PP_PASTE(A, DYND_PP_PASTE(_, DYND_PP_PASTE(B, \
    DYND_PP_PASTE(_, DYND_PP_PASTE(C, DYND_PP_PASTE(_, D))))))))
#define DYND_PP__IS_NULL_0_0_0_0 DYND_PP__IS_NULL_0_0_0_0
#define DYND_PP__IS_NULL_0_0_0_1 ,
#define DYND_PP__IS_NULL_0_0_1_0 DYND_PP__IS_NULL_0_0_1_0
#define DYND_PP__IS_NULL_0_0_1_1 DYND_PP__IS_NULL_0_0_1_1
#define DYND_PP__IS_NULL_0_1_0_0 DYND_PP__IS_NULL_0_1_0_0
#define DYND_PP__IS_NULL_0_1_0_1 DYND_PP__IS_NULL_0_1_0_1
#define DYND_PP__IS_NULL_0_1_1_0 DYND_PP__IS_NULL_0_1_1_0
#define DYND_PP__IS_NULL_0_1_1_1 DYND_PP__IS_NULL_0_1_1_1
#define DYND_PP__IS_NULL_1_0_0_0 DYND_PP__IS_NULL_1_0_0_0
#define DYND_PP__IS_NULL_1_0_0_1 DYND_PP__IS_NULL_1_0_0_1
#define DYND_PP__IS_NULL_1_0_1_0 DYND_PP__IS_NULL_1_0_1_0
#define DYND_PP__IS_NULL_1_0_1_1 DYND_PP__IS_NULL_1_0_1_1
#define DYND_PP__IS_NULL_1_1_0_0 DYND_PP__IS_NULL_1_1_0_0
#define DYND_PP__IS_NULL_1_1_0_1 DYND_PP__IS_NULL_1_1_0_1
#define DYND_PP__IS_NULL_1_1_1_0 DYND_PP__IS_NULL_1_1_1_0
#define DYND_PP__IS_NULL_1_1_1_1 DYND_PP__IS_NULL_1_1_1_1

#endif
