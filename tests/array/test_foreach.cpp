//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <inc_gtest.hpp>

#include <dynd/array.hpp>
#include <dynd/foreach.hpp>

using namespace std;
using namespace dynd;

int intfunc(int x, int y)
{
    return 2 * (x - y);
}

TEST(ArrayViews, IntFunc) {
    nd::array a = 10, b = 20, c;

    c = nd::foreach(a, b, intfunc);
    EXPECT_EQ(-20, c.as<int>());

    int aval0[2][3] = {{0, 1, 2}, {5, 6, 7}};
    int bval0[3] = {5, 2, 4};
    a = aval0;
    b = bval0;
    c = nd::foreach(a, b, intfunc);
    EXPECT_EQ(ndt::type("strided * strided * int32"), c.get_type());
    ASSERT_EQ(2, c.get_shape()[0]);
    ASSERT_EQ(3, c.get_shape()[1]);
    EXPECT_EQ(-10, c(0,0).as<int>());
    EXPECT_EQ(-2, c(0,1).as<int>());
    EXPECT_EQ(-4, c(0,2).as<int>());
    EXPECT_EQ(0, c(1,0).as<int>());
    EXPECT_EQ(8, c(1,1).as<int>());
    EXPECT_EQ(6, c(1,2).as<int>());
}

void vecintfunc1(int (&out)[1], int x, int y, int z)
{
    out[0] = x + y + z;
}


void vecintfunc(int (&out)[2], int x, int y)
{
    out[0] = y;
    out[1] = x;
}

void vecintfunc2(int (&out)[3], int x, int y, int z)
{
    out[0] = y;
    out[1] = x;
    out[2] = z;
}

TEST(ArrayViews, VecIntFunc) {
    nd::array a = 10, b = 20, c;

    c = nd::foreach(a, b, vecintfunc);
    EXPECT_EQ(20, c(0).as<int>());
    EXPECT_EQ(10, c(1).as<int>());

    int aval0[2][3] = {{0, 1, 2}, {5, 6, 7}};
    int bval0[3] = {5, 2, 4};
    a = aval0;
    b = bval0;
    c = nd::foreach(a, b, vecintfunc);

    EXPECT_EQ(ndt::type("strided * strided * 2 * int32"), c.get_type());
    ASSERT_EQ(2, c.get_shape()[0]);
    ASSERT_EQ(3, c.get_shape()[1]);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(bval0[j], c(i,j,0).as<int>());
            EXPECT_EQ(aval0[i][j], c(i,j,1).as<int>());
        }
    }

    c = nd::foreach(a, b, b, vecintfunc2);
    EXPECT_EQ(ndt::type("strided * strided * 3 * int32"), c.get_type());
    ASSERT_EQ(2, c.get_shape()[0]);
    ASSERT_EQ(3, c.get_shape()[1]);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(bval0[j], c(i,j,0).as<int>());
            EXPECT_EQ(aval0[i][j], c(i,j,1).as<int>());
            EXPECT_EQ(bval0[j], c(i,j,2).as<int>());
        }
    }
}

class IntMemFuncWrapper {
public:
    int operator ()(int x, int y) {
        return 2 * (x - y);
    }
};

TEST(ArrayViews, IntMemFunc) {
    nd::array a = 10, b = 20, c;

    c = nd::foreach(a, b, IntMemFuncWrapper());
    EXPECT_EQ(-20, c.as<int>());

    int aval0[2][3] = {{0, 1, 2}, {5, 6, 7}};
    int bval0[3] = {5, 2, 4};
    a = aval0;
    b = bval0;
    c = nd::foreach(a, b, IntMemFuncWrapper());
    EXPECT_EQ(ndt::type("strided * strided * int32"), c.get_type());
    ASSERT_EQ(2, c.get_shape()[0]);
    ASSERT_EQ(3, c.get_shape()[1]);
    EXPECT_EQ(-10, c(0,0).as<int>());
    EXPECT_EQ(-2, c(0,1).as<int>());
    EXPECT_EQ(-4, c(0,2).as<int>());
    EXPECT_EQ(0, c(1,0).as<int>());
    EXPECT_EQ(8, c(1,1).as<int>());
    EXPECT_EQ(6, c(1,2).as<int>());
}
