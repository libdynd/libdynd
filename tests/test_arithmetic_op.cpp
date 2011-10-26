#include <iostream>
#include <stdexcept>
#include <cmath>
#include <stdint.h>
#include <inc_gtest.hpp>

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
    c = (a + b).vals();
    EXPECT_EQ(make_dtype<int>(), c.get_dtype());
    EXPECT_EQ(1, c(0,0).as<int>());
    EXPECT_EQ(3, c(0,1).as<int>());
    EXPECT_EQ(4, c(0,2).as<int>());
    EXPECT_EQ(3, c(1,0).as<int>());
    EXPECT_EQ(7, c(1,1).as<int>());
    EXPECT_EQ(-7, c(1,2).as<int>());
    c = (a - b).vals();
    EXPECT_EQ(make_dtype<int>(), c.get_dtype());
    EXPECT_EQ(1, c(0,0).as<int>());
    EXPECT_EQ(1, c(0,1).as<int>());
    EXPECT_EQ(2, c(0,2).as<int>());
    EXPECT_EQ(-1, c(1,0).as<int>());
    EXPECT_EQ(-3, c(1,1).as<int>());
    EXPECT_EQ(13, c(1,2).as<int>());
    c = (b * a).vals();
    EXPECT_EQ(make_dtype<int>(), c.get_dtype());
    EXPECT_EQ(0, c(0,0).as<int>());
    EXPECT_EQ(2, c(0,1).as<int>());
    EXPECT_EQ(3, c(0,2).as<int>());
    EXPECT_EQ(2, c(1,0).as<int>());
    EXPECT_EQ(10, c(1,1).as<int>());
    EXPECT_EQ(-30, c(1,2).as<int>());
    c = (b / a).vals();
    EXPECT_EQ(make_dtype<int>(), c.get_dtype());
    EXPECT_EQ(0, c(0,0).as<int>());
    EXPECT_EQ(0, c(0,1).as<int>());
    EXPECT_EQ(0, c(0,2).as<int>());
    EXPECT_EQ(2, c(1,0).as<int>());
    EXPECT_EQ(2, c(1,1).as<int>());
    EXPECT_EQ(-3, c(1,2).as<int>());

    // A scalar on the right
    c = (a + 12).vals();
    EXPECT_EQ(13, c(0).as<int>());
    EXPECT_EQ(14, c(1).as<int>());
    EXPECT_EQ(15, c(2).as<int>());
    c = (a - 12).vals();
    EXPECT_EQ(-11, c(0).as<int>());
    EXPECT_EQ(-10, c(1).as<int>());
    EXPECT_EQ(-9, c(2).as<int>());
    c = (a * 3).vals();
    EXPECT_EQ(3, c(0).as<int>());
    EXPECT_EQ(6, c(1).as<int>());
    EXPECT_EQ(9, c(2).as<int>());
    c = (a / 2).vals();
    EXPECT_EQ(0, c(0).as<int>());
    EXPECT_EQ(1, c(1).as<int>());
    EXPECT_EQ(1, c(2).as<int>());

    // A scalar on the left
    c = ((-1) + a).vals();
    EXPECT_EQ(0, c(0).as<int>());
    EXPECT_EQ(1, c(1).as<int>());
    EXPECT_EQ(2, c(2).as<int>());
    c = ((-1) - a).vals();
    EXPECT_EQ(-2, c(0).as<int>());
    EXPECT_EQ(-3, c(1).as<int>());
    EXPECT_EQ(-4, c(2).as<int>());
    c = (5 * a).vals();
    EXPECT_EQ(5, c(0).as<int>());
    EXPECT_EQ(10, c(1).as<int>());
    EXPECT_EQ(15, c(2).as<int>());
    c = (-6 / a).vals();
    EXPECT_EQ(-6, c(0).as<int>());
    EXPECT_EQ(-3, c(1).as<int>());
    EXPECT_EQ(-2, c(2).as<int>());
}

TEST(ArithmeticOp, MatchingDTypes_View) {
    ndarray a, b, c, d;

    // Two arrays with broadcasting
    int v0[] = {1,2,3};
    int v1[][3] = {{0,1,1}, {2,5,-10}};
    a = v0;
    b = v1;

    c = a + b;
    EXPECT_EQ(1, c(0,0).as<int>());
    EXPECT_EQ(3, c(0,1).as<int>());
    EXPECT_EQ(4, c(0,2).as<int>());
    EXPECT_EQ(3, c(1,0).as<int>());
    EXPECT_EQ(7, c(1,1).as<int>());
    EXPECT_EQ(-7, c(1,2).as<int>());

    // Note: 'c' contains an expression tree with an 'add' node,
    // so editing the values of 'a' or 'b' changes the values of 'c'
    int v2[] = {6,4,2};
    a.vassign(ndarray(v2));
    EXPECT_EQ(6, c(0,0).as<int>());
    EXPECT_EQ(5, c(0,1).as<int>());
    EXPECT_EQ(3, c(0,2).as<int>());
    EXPECT_EQ(8, c(1,0).as<int>());
    EXPECT_EQ(9, c(1,1).as<int>());
    EXPECT_EQ(-8, c(1,2).as<int>());

    // Check also partial indexing through another temporary
    d = c(0);
    EXPECT_EQ(1, d.get_ndim());
    EXPECT_EQ(3, d.get_shape(0));
    a.vassign(ndarray(v0));
    EXPECT_EQ(1, d(0).as<int>());
    EXPECT_EQ(3, d(1).as<int>());
    EXPECT_EQ(4, d(2).as<int>());
    d = c(1);
    a.vassign(ndarray(v2));
    EXPECT_EQ(8, d(0).as<int>());
    EXPECT_EQ(9, d(1).as<int>());
    EXPECT_EQ(-8, d(2).as<int>());
}
