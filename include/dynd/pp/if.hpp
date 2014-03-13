//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__IF_HPP_
#define _DYND__IF_HPP_

#include <dynd/pp/gen.hpp>
#include <dynd/pp/logical.hpp>
#include <dynd/pp/token.hpp>

#define DYND_PP_IF(CONDITION) DYND_PP_CAT_2(DYND_PP_IF_, DYND_PP_BOOL(CONDITION))
#define DYND_PP_IF_0(...)
#define DYND_PP_IF_1(...) __VA_ARGS__

#define DYND_PP_IF_ELSE(CONDITION) DYND_PP_CAT_2(DYND_PP_IF_ELSE_, DYND_PP_BOOL(CONDITION))
#define DYND_PP_IF_ELSE_0(...) DYND_PP_ID
#define DYND_PP_IF_ELSE_1(...) __VA_ARGS__ DYND_PP_NULL

#endif
