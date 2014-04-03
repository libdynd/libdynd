//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "inc_gtest.hpp"

#include <dynd/pp/logical.hpp>

using namespace std;

TEST(PPLogical, Bool) {
    EXPECT_FALSE(DYND_PP_BOOL(0));
    EXPECT_TRUE(DYND_PP_BOOL(1));
    EXPECT_TRUE(DYND_PP_BOOL(3));
    EXPECT_TRUE(DYND_PP_BOOL(7));
    EXPECT_TRUE(DYND_PP_BOOL(DYND_PP_INT_MAX));
    EXPECT_TRUE(DYND_PP_BOOL(DYND_PP_LEN_MAX));
}

TEST(PPLogical, Not) {
    EXPECT_TRUE(DYND_PP_NOT(0));
    EXPECT_FALSE(DYND_PP_NOT(1));
}

TEST(PPLogical, And) {
    EXPECT_FALSE(DYND_PP_AND(0, 0));
    EXPECT_FALSE(DYND_PP_AND(0, 1));
    EXPECT_FALSE(DYND_PP_AND(1, 0));
    EXPECT_TRUE(DYND_PP_AND(1, 1));
}

TEST(PPLogical, Or) {
    EXPECT_FALSE(DYND_PP_OR(0, 0));
    EXPECT_TRUE(DYND_PP_OR(0, 1));
    EXPECT_TRUE(DYND_PP_OR(1, 0));
    EXPECT_TRUE(DYND_PP_OR(1, 1));
}

TEST(PPLogical, Xor) {
    EXPECT_FALSE(DYND_PP_XOR(0, 0));
    EXPECT_TRUE(DYND_PP_XOR(0, 1));
    EXPECT_TRUE(DYND_PP_XOR(1, 0));
    EXPECT_FALSE(DYND_PP_XOR(1, 1));
}
