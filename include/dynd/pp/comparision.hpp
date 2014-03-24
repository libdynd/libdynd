//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__COMPARISION_HPP_
#define _DYND__COMPARISION_HPP_

#include <dynd/pp/gen.hpp>

#include "C:\Users\Irwin\Repositories\libdynd\include\dynd\pp\token.hpp"
#include "C:\Users\Irwin\Repositories\libdynd\include\dynd\pp\logical.hpp"

#define DYND_PP_LT(A, B) DYND_PP__LT(DYND_PP_EQ(B, 0), DYND_PP_GET(A, \
    DYND_PP_SLICE_TO(B, DYND_PP_ONES_63), DYND_PP_ZEROS_63))
#define DYND_PP__LT(FLAG, VALUE) DYND_PP_CAT_2(DYND_PP__LT_, FLAG)(VALUE)
#define DYND_PP__LT_0(VALUE) VALUE
#define DYND_PP__LT_1(VALUE) 0

#define DYND_PP_EQ(A, B) DYND_PP_HAS_COMMA(DYND_PP_CAT_4(DYND_PP_EQ_, A, _, B))

#define DYND_PP_NE(A, B) DYND_PP_NOT(DYND_PP_EQ(A, B))

#define DYND_PP_GT(A, B) DYND_PP_LT(B, A)

#endif
