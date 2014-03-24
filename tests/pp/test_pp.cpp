//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "inc_gtest.hpp"

#include <dynd/pp/arithmetic.hpp>
#include <dynd/pp/comparision.hpp>
#include <dynd/pp/if.hpp>
#include <dynd/pp/list.hpp>
#include <dynd/pp/logical.hpp>

//#include "C:\Users\Irwin\Repositories\libdynd\include\dynd\pp\arithmetic.hpp"
//#include "C:\Users\Irwin\Repositories\libdynd\include\dynd\pp\comparision.hpp"

using namespace std;

TEST(PP, Cat) {
    EXPECT_EQ(DYND_PP_CAT(1, 2, 3), 123);
    EXPECT_EQ(DYND_PP_CAT(1, 2, 3, 4), 1234);
}

TEST(PP, Comparision) {
    EXPECT_TRUE(DYND_PP_EQ(0, 0));

//    EXPECT_TRUE(DYND_PP_LT(0, 1));
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

TEST(PP, Arithmetic) {
    EXPECT_EQ(DYND_PP_INC(0), 1);
    EXPECT_EQ(DYND_PP_INC(6), 7);
    EXPECT_EQ(DYND_PP_INC(DYND_PP_DEC(DYND_PP_INT_MAX)), DYND_PP_INT_MAX);

    EXPECT_EQ(DYND_PP_DEC(1), 0);
    EXPECT_EQ(DYND_PP_DEC(7), 6);
    EXPECT_EQ(DYND_PP_DEC(DYND_PP_INT_MAX), DYND_PP_INT_MAX - 1);

    EXPECT_EQ(DYND_PP_ADD(0, 0), 0);
    EXPECT_EQ(DYND_PP_ADD(0, 1), 1);
    EXPECT_EQ(DYND_PP_ADD(1, 0), 1);
    EXPECT_EQ(DYND_PP_ADD(4, 2), 6);
    EXPECT_EQ(DYND_PP_ADD(3, 4), 7);
    EXPECT_EQ(DYND_PP_ADD(4, 3), 7);
    EXPECT_EQ(DYND_PP_ADD(0, DYND_PP_INT_MAX), DYND_PP_INT_MAX);
    EXPECT_EQ(DYND_PP_ADD(DYND_PP_INT_MAX, 0), DYND_PP_INT_MAX);

    EXPECT_EQ(DYND_PP_SUB(0, 0), 0);
    EXPECT_EQ(DYND_PP_SUB(1, 0), 1);
    EXPECT_EQ(DYND_PP_SUB(4, 2), 2);
    EXPECT_EQ(DYND_PP_SUB(4, 3), 1);
    EXPECT_EQ(DYND_PP_SUB(DYND_PP_INT_MAX, 0), DYND_PP_INT_MAX);
}

TEST(PP, If) {
    EXPECT_FALSE(DYND_PP_IS_EMPTY(DYND_PP_IF(1)(A)));
    EXPECT_TRUE(DYND_PP_IS_EMPTY(DYND_PP_IF(0)(A)));
}
