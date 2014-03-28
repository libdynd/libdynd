//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "inc_gtest.hpp"

#include <dynd/pp/token.hpp>

using namespace std;

TEST(PPToken, IsNull) {
    EXPECT_TRUE(DYND_PP_IS_NULL());

    EXPECT_FALSE(DYND_PP_IS_NULL(A));
    EXPECT_FALSE(DYND_PP_IS_NULL((A)));
    EXPECT_FALSE(DYND_PP_IS_NULL(0));
    EXPECT_FALSE(DYND_PP_IS_NULL((0)));

    EXPECT_FALSE(DYND_PP_IS_NULL(,));
    EXPECT_FALSE(DYND_PP_IS_NULL(,));
    EXPECT_FALSE(DYND_PP_IS_NULL(A, B));
    EXPECT_FALSE(DYND_PP_IS_NULL((A, B)));
    EXPECT_FALSE(DYND_PP_IS_NULL(A + B));
    EXPECT_FALSE(DYND_PP_IS_NULL(0, 1));
    EXPECT_FALSE(DYND_PP_IS_NULL((0, 1)));

    EXPECT_FALSE(DYND_PP_IS_NULL(,,,,,,,));
    EXPECT_FALSE(DYND_PP_IS_NULL(A, B, C, D, E, F, G, H));
    EXPECT_FALSE(DYND_PP_IS_NULL((A, B, C, D, E, F, G, H)));
    EXPECT_FALSE(DYND_PP_IS_NULL(A + B + C + D + E + F + G + H));
    EXPECT_FALSE(DYND_PP_IS_NULL(0, 1, 2, 3, 4, 5, 6, 7));
    EXPECT_FALSE(DYND_PP_IS_NULL((0, 1, 2, 3, 4, 5, 6, 7)));
}
