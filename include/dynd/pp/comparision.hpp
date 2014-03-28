//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__COMPARISION_HPP_
#define _DYND__COMPARISION_HPP_

#include <dynd/pp/gen.hpp>
#include <dynd/pp/list.hpp>
#include <dynd/pp/logical.hpp>
#include <dynd/pp/token.hpp>

#define DYND_PP_LT(A, B) DYND_PP_PASTE(DYND_PP_LT_, DYND_PP_PASTE(DYND_PP_BOOL(A), \
    DYND_PP_PASTE(_, DYND_PP_BOOL(B))))(DYND_PP_GET(A, (DYND_PP_ID DYND_PP_SLICE_TO(B, \
    DYND_PP_PASTE(DYND_PP_ONES_, DYND_PP_LEN_MAX)), DYND_PP_ID DYND_PP_PASTE(DYND_PP_ZEROS_, \
    DYND_PP_LEN_MAX))))
#define DYND_PP_LT_0_0(...) 0
#define DYND_PP_LT_1_0(...) 0
#define DYND_PP_LT_0_1 DYND_PP_ID
#define DYND_PP_LT_1_1 DYND_PP_ID

#define DYND_PP_LE(A, B) DYND_PP_GE(B, A)

#define DYND_PP_EQ(A, B) DYND_PP_IS_NULL(DYND_PP_PASTE(DYND_PP_, DYND_PP_PASTE(A, DYND_PP_PASTE(_EQ_, B))))

#define DYND_PP_NE(A, B) DYND_PP_NOT(DYND_PP_EQ(A, B))

#define DYND_PP_GE(A, B) DYND_PP_PASTE(DYND_PP_GE_, DYND_PP_PASTE(DYND_PP_BOOL(A), \
    DYND_PP_PASTE(_, DYND_PP_BOOL(B))))(DYND_PP_GET(A, (DYND_PP_ID DYND_PP_SLICE_TO(B, \
    DYND_PP_PASTE(DYND_PP_ZEROS_, DYND_PP_LEN_MAX)), DYND_PP_ID DYND_PP_PASTE(DYND_PP_ONES_, \
    DYND_PP_LEN_MAX))))
#define DYND_PP_GE_0_0(...) 1
#define DYND_PP_GE_0_1 DYND_PP_ID
#define DYND_PP_GE_1_0(...) 1
#define DYND_PP_GE_1_1 DYND_PP_ID

#define DYND_PP_GT(A, B) DYND_PP_LT(B, A)

#endif
