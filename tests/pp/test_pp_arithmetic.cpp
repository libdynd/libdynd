//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "inc_gtest.hpp"

#include <dynd/pp/arithmetic.hpp>

using namespace std;

TEST(PPArithmetic, Inc) {
    EXPECT_EQ(DYND_PP_INC(0), 1);
    EXPECT_EQ(DYND_PP_INC(3), 4);
    EXPECT_EQ(DYND_PP_INC(6), 7);
    EXPECT_EQ(DYND_PP_INC(DYND_PP_DEC(DYND_PP_INT_MAX)), DYND_PP_INT_MAX);
}

TEST(PPArithmetic, Dec) {
    EXPECT_EQ(DYND_PP_DEC(1), 0);
    EXPECT_EQ(DYND_PP_DEC(4), 3);
    EXPECT_EQ(DYND_PP_DEC(7), 6);
    EXPECT_EQ(DYND_PP_DEC(DYND_PP_INT_MAX), DYND_PP_INT_MAX - 1);
}

TEST(PPArithmetic, Add) {
    EXPECT_EQ(DYND_PP_ADD(0, 0), 0);
    EXPECT_EQ(DYND_PP_ADD(0, 1), 1);
    EXPECT_EQ(DYND_PP_ADD(1, 0), 1);
    EXPECT_EQ(DYND_PP_ADD(4, 2), 6);
    EXPECT_EQ(DYND_PP_ADD(3, 4), 7);
    EXPECT_EQ(DYND_PP_ADD(4, 3), 7);
    EXPECT_EQ(DYND_PP_ADD(0, 7), 7);
    EXPECT_EQ(DYND_PP_ADD(7, 0), 7);
    EXPECT_EQ(DYND_PP_ADD(0, DYND_PP_INT_MAX), DYND_PP_INT_MAX);
    EXPECT_EQ(DYND_PP_ADD(DYND_PP_INT_MAX, 0), DYND_PP_INT_MAX);
}

TEST(PPArithmetic, Sub) {
    EXPECT_EQ(DYND_PP_SUB(0, 0), 0);
    EXPECT_EQ(DYND_PP_SUB(1, 0), 1);
    EXPECT_EQ(DYND_PP_SUB(4, 2), 2);
    EXPECT_EQ(DYND_PP_SUB(4, 3), 1);
    EXPECT_EQ(DYND_PP_SUB(7, 0), 7);
    EXPECT_EQ(DYND_PP_SUB(DYND_PP_INT_MAX, 0), DYND_PP_INT_MAX);
}
