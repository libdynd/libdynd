//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__PP_LOGICAL_HPP_
#define _DYND__PP_LOGICAL_HPP_

#include <dynd/pp/comparison.hpp>
#include <dynd/pp/token.hpp>

#define DYND_PP_BOOL(A) DYND_PP_PASTE(DYND_PP_BOOL_, DYND_PP_EQ(A, 0))
#define DYND_PP_BOOL_0 1
#define DYND_PP_BOOL_1 0

#define DYND_PP_NOT(A) DYND_PP_PASTE(DYND_PP_NOT_, DYND_PP_BOOL(A))
#define DYND_PP_NOT_0 1
#define DYND_PP_NOT_1 0

#define DYND_PP_AND(A, B) DYND_PP_PASTE(DYND_PP_AND_, DYND_PP_PASTE(DYND_PP_BOOL(A), DYND_PP_PASTE(_, DYND_PP_BOOL(B))))
#define DYND_PP_AND_0_0 0
#define DYND_PP_AND_0_1 0
#define DYND_PP_AND_1_0 0
#define DYND_PP_AND_1_1 1

#define DYND_PP_OR(A, B) DYND_PP_PASTE(DYND_PP_OR_, DYND_PP_PASTE(DYND_PP_BOOL(A), DYND_PP_PASTE(_, DYND_PP_BOOL(B))))
#define DYND_PP_OR_0_0 0
#define DYND_PP_OR_0_1 1
#define DYND_PP_OR_1_0 1
#define DYND_PP_OR_1_1 1

#define DYND_PP_XOR(A, B) DYND_PP_EQ(DYND_PP_NOT(A), DYND_PP_BOOL(B))

#endif
