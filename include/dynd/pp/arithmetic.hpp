//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ARITHMETIC_HPP_
#define _DYND__ARITHMETIC_HPP_

#include <dynd/pp/gen.hpp>

//#include <dynd/pp/list.hpp>

#include "C:\Users\Irwin\Repositories\libdynd\include\dynd\pp\list.hpp"

#define DYND_PP_ADD(A, B) DYND_PP_LEN(DYND_PP_SLICE_TO(A, DYND_PP_CAT_2(DYND_PP_ZEROS_, DYND_PP_INT_MAX)) DYND_PP_IF(DYND_PP_AND(A, B))(,) \
    DYND_PP_SLICE_TO(B, DYND_PP_CAT_2(DYND_PP_ZEROS_, DYND_PP_INT_MAX)))
#define DYND_PP_SUB(A, B) DYND_PP_LEN(DYND_PP_SLICE_FROM(B, DYND_PP_SLICE_TO(A, DYND_PP_CAT_2(DYND_PP_ZEROS_, DYND_PP_INT_MAX))))

#endif
