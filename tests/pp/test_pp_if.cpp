//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "inc_gtest.hpp"

#include <dynd/pp/if.hpp>
#include <dynd/pp/token.hpp>

using namespace std;

TEST(PPIf, If) {
    EXPECT_TRUE(DYND_PP_IS_NULL(DYND_PP_IF(0)(3)));
    EXPECT_TRUE(DYND_PP_IS_NULL(DYND_PP_IF(0)(A)));
    EXPECT_EQ(DYND_PP_IF(1)(3), 3);
    EXPECT_FALSE(DYND_PP_IS_NULL(DYND_PP_IF(1)(A)));
}

TEST(PPIf, IfElse) {
    EXPECT_EQ(DYND_PP_IF_ELSE(0)(3)(7), 7);
    EXPECT_FALSE(DYND_PP_IS_NULL(DYND_PP_IF_ELSE(0)(A)(B)));
    EXPECT_EQ(DYND_PP_IF_ELSE(1)(3)(7), 3);
    EXPECT_FALSE(DYND_PP_IS_NULL(DYND_PP_IF_ELSE(1)(A)(B)));
}
