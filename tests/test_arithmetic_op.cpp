#include <iostream>
#include <stdexcept>
#include <cmath>
#include <stdint.h>
#include <gtest/gtest.h>

#include "dnd/ndarray.hpp"

using namespace std;
using namespace dnd;

TEST(ArithmeticOp, MatchingDTypes) {
    ndarray a, b, c;

    // Two arrays with broadcasting
    int v0[] = {1,2,3};
    int v1[][3] = {{0,1,1}, {2,5,-10}};
    a = v0;
    b = v1;
    c = (a + b).as_strided();
    EXPECT_EQ(make_dtype<int>(), c.get_dtype());
    EXPECT_EQ(1, c(0,0).as<int>());
    EXPECT_EQ(3, c(0,1).as<int>());
    EXPECT_EQ(4, c(0,2).as<int>());
    EXPECT_EQ(3, c(1,0).as<int>());
    EXPECT_EQ(7, c(1,1).as<int>());
    EXPECT_EQ(-7, c(1,2).as<int>());
    c = (a - b).as_strided();
    EXPECT_EQ(make_dtype<int>(), c.get_dtype());
    EXPECT_EQ(1, c(0,0).as<int>());
    EXPECT_EQ(1, c(0,1).as<int>());
    EXPECT_EQ(2, c(0,2).as<int>());
    EXPECT_EQ(-1, c(1,0).as<int>());
    EXPECT_EQ(-3, c(1,1).as<int>());
    EXPECT_EQ(13, c(1,2).as<int>());
    c = (b * a).as_strided();
    EXPECT_EQ(make_dtype<int>(), c.get_dtype());
    EXPECT_EQ(0, c(0,0).as<int>());
    EXPECT_EQ(2, c(0,1).as<int>());
    EXPECT_EQ(3, c(0,2).as<int>());
    EXPECT_EQ(2, c(1,0).as<int>());
    EXPECT_EQ(10, c(1,1).as<int>());
    EXPECT_EQ(-30, c(1,2).as<int>());
    c = (b / a).as_strided();
    EXPECT_EQ(make_dtype<int>(), c.get_dtype());
    EXPECT_EQ(0, c(0,0).as<int>());
    EXPECT_EQ(0, c(0,1).as<int>());
    EXPECT_EQ(0, c(0,2).as<int>());
    EXPECT_EQ(2, c(1,0).as<int>());
    EXPECT_EQ(2, c(1,1).as<int>());
    EXPECT_EQ(-3, c(1,2).as<int>());

    // A scalar on the right
    c = (a + 12).as_strided();
    EXPECT_EQ(13, c(0).as<int>());
    EXPECT_EQ(14, c(1).as<int>());
    EXPECT_EQ(15, c(2).as<int>());
    c = (a - 12).as_strided();
    EXPECT_EQ(-11, c(0).as<int>());
    EXPECT_EQ(-10, c(1).as<int>());
    EXPECT_EQ(-9, c(2).as<int>());
    c = (a * 3).as_strided();
    EXPECT_EQ(3, c(0).as<int>());
    EXPECT_EQ(6, c(1).as<int>());
    EXPECT_EQ(9, c(2).as<int>());
    c = (a / 2).as_strided();
    EXPECT_EQ(0, c(0).as<int>());
    EXPECT_EQ(1, c(1).as<int>());
    EXPECT_EQ(1, c(2).as<int>());

    // A scalar on the left
    c = ((-1) + a).as_strided();
    EXPECT_EQ(0, c(0).as<int>());
    EXPECT_EQ(1, c(1).as<int>());
    EXPECT_EQ(2, c(2).as<int>());
    c = ((-1) - a).as_strided();
    EXPECT_EQ(-2, c(0).as<int>());
    EXPECT_EQ(-3, c(1).as<int>());
    EXPECT_EQ(-4, c(2).as<int>());
    c = (5 * a).as_strided();
    EXPECT_EQ(5, c(0).as<int>());
    EXPECT_EQ(10, c(1).as<int>());
    EXPECT_EQ(15, c(2).as<int>());
    c = (-6 / a).as_strided();
    EXPECT_EQ(-6, c(0).as<int>());
    EXPECT_EQ(-3, c(1).as<int>());
    EXPECT_EQ(-2, c(2).as<int>());

    //(a + b).debug_dump(cout);
    //(a - b).debug_dump(cout);
    //(a * 2).debug_dump(cout);
    //(b / 5).debug_dump(cout);
}
