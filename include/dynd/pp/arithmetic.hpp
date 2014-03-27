//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ARITHMETIC_HPP_
#define _DYND__ARITHMETIC_HPP_

#include <dynd/pp/gen.hpp>
#include <dynd/pp/list.hpp>
#include <dynd/pp/logical.hpp>

#define DYND_PP_ADD(A, B) DYND_PP_LEN(DYND_PP_MERGE(DYND_PP_SLICE_TO(A, DYND_PP_CAT_2(DYND_PP_ZEROS_, DYND_PP_LEN_MAX)), \
    DYND_PP_SLICE_TO(B, DYND_PP_CAT_2(DYND_PP_ZEROS_, DYND_PP_LEN_MAX))))

#define DYND_PP_SUB(A, B) DYND_PP_LEN(DYND_PP_SLICE_FROM(B, DYND_PP_SLICE_TO(A, DYND_PP_CAT_2(DYND_PP_ZEROS_, DYND_PP_LEN_MAX))))

#endif
