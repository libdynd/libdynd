//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__COMPARISION_HPP_
#define _DYND__COMPARISION_HPP_

#include <dynd/pp/gen.hpp>
#include <dynd/pp/token.hpp>

#define DYND_PP_EQ(A, B) DYND_PP_HAS_COMMA(DYND_PP_CAT_4(DYND_PP_EQ_, A, _, B))

#endif
