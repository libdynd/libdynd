//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__TOKEN_HPP_
#define _DYND__TOKEN_HPP_

#include <dynd/pp/gen.hpp>

#define DYND_PP__MSC_EVAL(...) __VA_ARGS__

#define DYND_PP_TO_COMMA(...) ,

#define DYND_PP_CAT(...) DYND_PP_APPLY(DYND_PP_CAT_2(DYND_PP_CAT_, DYND_PP_ID(DYND_PP_LEN(__VA_ARGS__))), (__VA_ARGS__))
#define DYND_PP_CAT_0()
#define DYND_PP_CAT_1(A) A
//#define DYND_PP_CAT_2(A, ...) DYND_PP__CAT_2(A, __VA_ARGS__)
//#define DYND_PP__CAT_2(A, ...) DYND_PP___CAT_2(A, __VA_ARGS__)
//#define DYND_PP___CAT_2(A, B) A ## B

#endif
