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
    EXPECT_EQ(DYND_PP_LEN(,), 2);

    EXPECT_EQ(DYND_PP_LEN(A, B, C, D, E, F, G, H), 8);
    EXPECT_EQ(DYND_PP_LEN(,,,,,,,), 8);

    EXPECT_EQ(DYND_PP_LEN(DYND_PP_INTS), DYND_PP_LEN_MAX);
}

TEST(PPList, AllEq) {
//    EXPECT_TRUE(DYND_PP_ALL_EQ((), ()));

    EXPECT_TRUE(DYND_PP_ALL_EQ((0), (0)));
    EXPECT_FALSE(DYND_PP_ALL_EQ((0), (1)));

//    EXPECT_FALSE(DYND_PP_ALL_EQ((), (0)));

    EXPECT_TRUE(DYND_PP_ALL_EQ((0, 1), (0, 1)));
    EXPECT_FALSE(DYND_PP_ALL_EQ((0, 1), (1, 0)));

//    EXPECT_FALSE(DYND_PP_ALL_EQ((0), (0, 1)));

    EXPECT_TRUE(DYND_PP_ALL_EQ((5, 1, 4, 3, 7, 0, 2, 6), (5, 1, 4, 3, 7, 0, 2, 6)));
    EXPECT_FALSE(DYND_PP_ALL_EQ((5, 1, 4, 3, 7, 0, 2, 6), (5, 1, 4, 3, 3, 0, 2, 6)));
    EXPECT_FALSE(DYND_PP_ALL_EQ((5, 1, 4, 3, 7, 0, 2, 6), (6, 2, 0, 7, 3, 4, 1, 5)));

//    EXPECT_FALSE(DYND_PP_ALL_EQ((0, 1), (6, 2, 0, 7, 3, 4, 1, 5)));
}



TEST(PPList, Range) {
//    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_RANGE(0)), ()));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_RANGE(1)), (0)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_RANGE(2)), (0, 1)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_RANGE(8)), (0, 1, 2, 3, 4, 5, 6, 7)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_RANGE(DYND_PP_LEN_MAX)), (DYND_PP_INTS)));
}

TEST(PPList, Get) {
    EXPECT_EQ(DYND_PP_GET(0, 0), 0);

    EXPECT_EQ(DYND_PP_GET(0, 0, 1), 0);
    EXPECT_EQ(DYND_PP_GET(1, 0, 1), 1);

    EXPECT_EQ(DYND_PP_GET(0, 5, 1, 4, 3, 7, 0, 2, 6), 5);
    EXPECT_EQ(DYND_PP_GET(1, 5, 1, 4, 3, 7, 0, 2, 6), 1);
    EXPECT_EQ(DYND_PP_GET(2, 5, 1, 4, 3, 7, 0, 2, 6), 4);
    EXPECT_EQ(DYND_PP_GET(3, 5, 1, 4, 3, 7, 0, 2, 6), 3);
    EXPECT_EQ(DYND_PP_GET(4, 5, 1, 4, 3, 7, 0, 2, 6), 7);
    EXPECT_EQ(DYND_PP_GET(5, 5, 1, 4, 3, 7, 0, 2, 6), 0);
    EXPECT_EQ(DYND_PP_GET(6, 5, 1, 4, 3, 7, 0, 2, 6), 2);
    EXPECT_EQ(DYND_PP_GET(7, 5, 1, 4, 3, 7, 0, 2, 6), 6);

    EXPECT_EQ(DYND_PP_GET(0, DYND_PP_RANGE(DYND_PP_LEN_MAX)), 0);
    EXPECT_EQ(DYND_PP_GET(DYND_PP_INT_MAX, DYND_PP_RANGE(DYND_PP_LEN_MAX)), DYND_PP_INT_MAX);
}

TEST(PPList, Set) {
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_SET(0, 1, 0)), (1)));

    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_SET(0, 1, 0, 1)), (1, 1)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_SET(1, 0, 0, 1)), (0, 0)));

    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_SET(0, 6, 5, 1, 4, 3, 7, 0, 2, 6)), (6, 1, 4, 3, 7, 0, 2, 6)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_SET(1, 2, 5, 1, 4, 3, 7, 0, 2, 6)), (5, 2, 4, 3, 7, 0, 2, 6)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_SET(2, 0, 5, 1, 4, 3, 7, 0, 2, 6)), (5, 1, 0, 3, 7, 0, 2, 6)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_SET(3, 7, 5, 1, 4, 3, 7, 0, 2, 6)), (5, 1, 4, 7, 7, 0, 2, 6)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_SET(4, 3, 5, 1, 4, 3, 7, 0, 2, 6)), (5, 1, 4, 3, 3, 0, 2, 6)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_SET(5, 4, 5, 1, 4, 3, 7, 0, 2, 6)), (5, 1, 4, 3, 7, 4, 2, 6)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_SET(6, 1, 5, 1, 4, 3, 7, 0, 2, 6)), (5, 1, 4, 3, 7, 0, 1, 6)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_SET(7, 5, 5, 1, 4, 3, 7, 0, 2, 6)), (5, 1, 4, 3, 7, 0, 2, 5)));
}

TEST(PPList, Del) {
//    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_DEL(0, 0)), ()));

    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_DEL(0, 0, 1)), (1)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_DEL(1, 0, 1)), (0)));

    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_DEL(0, 5, 1, 4, 3, 7, 0, 2, 6)), (1, 4, 3, 7, 0, 2, 6)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_DEL(1, 5, 1, 4, 3, 7, 0, 2, 6)), (5, 4, 3, 7, 0, 2, 6)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_DEL(2, 5, 1, 4, 3, 7, 0, 2, 6)), (5, 1, 3, 7, 0, 2, 6)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_DEL(3, 5, 1, 4, 3, 7, 0, 2, 6)), (5, 1, 4, 7, 0, 2, 6)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_DEL(4, 5, 1, 4, 3, 7, 0, 2, 6)), (5, 1, 4, 3, 0, 2, 6)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_DEL(5, 5, 1, 4, 3, 7, 0, 2, 6)), (5, 1, 4, 3, 7, 2, 6)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_DEL(6, 5, 1, 4, 3, 7, 0, 2, 6)), (5, 1, 4, 3, 7, 0, 6)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_DEL(7, 5, 1, 4, 3, 7, 0, 2, 6)), (5, 1, 4, 3, 7, 0, 2)));

    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_DEL(0, DYND_PP_RANGE(DYND_PP_LEN_MAX))), (DYND_PP_RANGE(1, DYND_PP_LEN_MAX))));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_DEL(DYND_PP_INT_MAX, DYND_PP_RANGE(DYND_PP_LEN_MAX))), (DYND_PP_RANGE(DYND_PP_INT_MAX))));
}

TEST(PPList, First) {
    EXPECT_EQ(DYND_PP_FIRST(0), 0);

    EXPECT_EQ(DYND_PP_FIRST(0, 1), 0);

    EXPECT_EQ(DYND_PP_FIRST(5, 1, 4, 3, 7, 0, 2, 6), 5);

    EXPECT_EQ(DYND_PP_FIRST(DYND_PP_RANGE(DYND_PP_LEN_MAX)), 0);
}

TEST(PPList, Last) {
    EXPECT_EQ(DYND_PP_LAST(0), 0);

    EXPECT_EQ(DYND_PP_LAST(0, 1), 1);

    EXPECT_EQ(DYND_PP_LAST(5, 1, 4, 3, 7, 0, 2, 6), 6);

    EXPECT_EQ(DYND_PP_LAST(DYND_PP_RANGE(DYND_PP_LEN_MAX)), DYND_PP_INT_MAX);
}

TEST(PPList, All) {
    EXPECT_TRUE(DYND_PP_ALL(1, 2, 3));
    EXPECT_TRUE(DYND_PP_ALL(DYND_PP_MAP(DYND_PP_INC, (,), 1, 2, 3)));
}


TEST(PPList, Map) {
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_MAP(DYND_PP_ID, (,), 0)), (0)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_MAP(DYND_PP_ID, (,), 0, 1)), (0, 1)));
    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_MAP(DYND_PP_ID, (,), 5, 1, 4, 3, 7, 0, 2, 6)), (5, 1, 4, 3, 7, 0, 2, 6)));
//    EXPECT_TRUE(DYND_PP_ALL_EQ((DYND_PP_MAP(DYND_PP_ID, (,), DYND_PP_RANGE(DYND_PP_LEN_MAX))), DYND_PP_RANGE(DYND_PP_LEN_MAX)));

    EXPECT_EQ(DYND_PP_MAP(DYND_PP_ID, (+), 0, 1, 2, 3, 4, 5, 6), 21);
    EXPECT_EQ(DYND_PP_MAP(DYND_PP_INC, (+), 0, 1, 2, 3, 4, 5, 6), 28);
}

TEST(PPList, Reduce) {
    EXPECT_TRUE(DYND_PP_REDUCE(DYND_PP_AND, 1, 1, 1, 1, 1));
}
