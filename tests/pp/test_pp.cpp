//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "inc_gtest.hpp"

#include <dynd/pp/if.hpp>
#include <dynd/pp/list.hpp>
#include <dynd/pp/logical.hpp>

using namespace std;

TEST(PPList, IsEmpty) {
    EXPECT_TRUE(DYND_PP_IS_EMPTY());

    EXPECT_FALSE(DYND_PP_IS_EMPTY(,));
    EXPECT_FALSE(DYND_PP_IS_EMPTY(A));
    EXPECT_FALSE(DYND_PP_IS_EMPTY(A, B));
    EXPECT_FALSE(DYND_PP_IS_EMPTY(A, B, C, D, E, F, G));
}

TEST(PPList, Len) {
    EXPECT_EQ(DYND_PP_LEN(), 0);

    EXPECT_EQ(DYND_PP_LEN(A), 1);
    EXPECT_EQ(DYND_PP_LEN(A, B), 2);
    EXPECT_EQ(DYND_PP_LEN(A, B, C, D, E, F, G), 7);

    EXPECT_EQ(DYND_PP_LEN(,), 2);
    EXPECT_EQ(DYND_PP_LEN(,,,,,,), 7);
}

TEST(PPLogical, Bool) {
    EXPECT_TRUE(DYND_PP_BOOL(3));
    EXPECT_TRUE(DYND_PP_BOOL(DYND_PP_INT_MAX));

    EXPECT_FALSE(DYND_PP_BOOL(0));
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

TEST(PP, If) {
    EXPECT_FALSE(DYND_PP_IS_EMPTY(DYND_PP_IF(1)(A)));
    EXPECT_TRUE(DYND_PP_IS_EMPTY(DYND_PP_IF(0)(A)));
}
