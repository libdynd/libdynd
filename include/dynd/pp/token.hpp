//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__TOKEN_HPP_
#define _DYND__TOKEN_HPP_

#include <dynd/pp/gen.hpp>

#define DYND_PP_TO_COMMA(...) ,

#define DYND_PP_NULL(...)
#define DYND_PP_ID(...) __VA_ARGS__

#define DYND_PP_CAT(...) DYND_PP__CAT_2(DYND_PP_CAT_, DYND_PP_LEN(__VA_ARGS__))(__VA_ARGS__)
#define DYND_PP_CAT_0()
#define DYND_PP_CAT_1(A) A
#define DYND_PP_CAT_2(A, ...) DYND_PP__CAT_2(A, __VA_ARGS__)
#define DYND_PP__CAT_2(A, ...) DYND_PP___CAT_2(A, __VA_ARGS__)
#define DYND_PP___CAT_2(A, B) A ## B
#define DYND_PP_CAT_3(A, ...) DYND_PP_CAT_2(A, DYND_PP_CAT_2(__VA_ARGS__))
#define DYND_PP_CAT_4(A, ...) DYND_PP_CAT_2(A, DYND_PP_CAT_3(__VA_ARGS__))
#define DYND_PP_CAT_5(A, ...) DYND_PP_CAT_2(A, DYND_PP_CAT_4(__VA_ARGS__))
#define DYND_PP_CAT_6(A, ...) DYND_PP_CAT_2(A, DYND_PP_CAT_5(__VA_ARGS__))
#define DYND_PP_CAT_7(A, ...) DYND_PP_CAT_2(A, DYND_PP_CAT_6(__VA_ARGS__))

#endif
