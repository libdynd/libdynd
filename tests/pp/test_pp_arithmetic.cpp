//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "inc_gtest.hpp"

#include <dynd/pp/arithmetic.hpp>

using namespace std;

TEST(PPArithmetic, Increment) {
    EXPECT_EQ(DYND_PP_INC(0), 1);

    EXPECT_EQ(DYND_PP_INC(3), 4);

    EXPECT_EQ(DYND_PP_INC(7), 8);

    EXPECT_EQ(DYND_PP_INC(DYND_PP_DEC(DYND_PP_LEN_MAX)), DYND_PP_LEN_MAX);
}

TEST(PPArithmetic, Decrement) {
    EXPECT_EQ(DYND_PP_DEC(1), 0);

    EXPECT_EQ(DYND_PP_DEC(4), 3);

    EXPECT_EQ(DYND_PP_DEC(8), 7);

    EXPECT_EQ(DYND_PP_DEC(DYND_PP_LEN_MAX), DYND_PP_LEN_MAX - 1);
}

TEST(PPArithmetic, Addition) {
    EXPECT_EQ(DYND_PP_ADD(0, 0), 0);

    EXPECT_EQ(DYND_PP_ADD(0, 1), 1);
    EXPECT_EQ(DYND_PP_ADD(1, 0), 1);

    EXPECT_EQ(DYND_PP_ADD(4, 2), 6);

    EXPECT_EQ(DYND_PP_ADD(3, 4), 7);
    EXPECT_EQ(DYND_PP_ADD(4, 3), 7);

    EXPECT_EQ(DYND_PP_ADD(4, 4), 8);

    EXPECT_EQ(DYND_PP_ADD(0, 8), 8);
    EXPECT_EQ(DYND_PP_ADD(8, 0), 8);

    EXPECT_EQ(DYND_PP_ADD(0, DYND_PP_LEN_MAX), DYND_PP_LEN_MAX);
    EXPECT_EQ(DYND_PP_ADD(DYND_PP_LEN_MAX, 0), DYND_PP_LEN_MAX);
}

TEST(PPArithmetic, Subtraction) {
    EXPECT_EQ(DYND_PP_SUB(0, 0), 0);

    EXPECT_EQ(DYND_PP_SUB(1, 0), 1);

    EXPECT_EQ(DYND_PP_SUB(4, 2), 2);

    EXPECT_EQ(DYND_PP_SUB(4, 3), 1);

    EXPECT_EQ(DYND_PP_SUB(4, 4), 0);

    EXPECT_EQ(DYND_PP_SUB(8, 0), 8);

    EXPECT_EQ(DYND_PP_SUB(DYND_PP_LEN_MAX, 0), DYND_PP_LEN_MAX);
}
