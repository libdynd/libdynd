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
#include <dynd/elwise.hpp>

using namespace std;
using namespace dynd;




void func(int &res, int (&x)[3]) {
    res = x[0] + x[1] + x[2];
}

//void func(int &res, int (&x)[3], int (&y)[3]) {
  //  res = x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
//}


TEST(ArrayViews, FixedFunc) {
    nd::array a, b;

    int vals[3] = {2, 4, 6};
    a = nd::empty(ndt::make_fixed_dim(3, ndt::make_type<int>()));
    a.vals() = vals;

    b = nd::elwise(func, a);
    EXPECT_EQ(ndt::type("int32"), b.get_type());
    EXPECT_EQ(12, b.as<int>());
}

int intfunc(int &x, int &y)
{
    return 2 * (x - y);
}

void intfunc_other(int &out, int x, int y)
{
    out = 2 * (x - y);
}


TEST(ArrayViews, IntFunc) {
    nd::array a = 10, b = 20, c;

    c = nd::elwise(intfunc, a, b);
    EXPECT_EQ(-20, c.as<int>());

    int aval0[2][3] = {{0, 1, 2}, {5, 6, 7}};
    int bval0[3] = {5, 2, 4};
    a = aval0;
    b = bval0;
    c = nd::elwise(intfunc, a, b);
    EXPECT_EQ(ndt::type("strided * strided * int32"), c.get_type());
    ASSERT_EQ(2, c.get_shape()[0]);
    ASSERT_EQ(3, c.get_shape()[1]);
    EXPECT_EQ(-10, c(0,0).as<int>());
    EXPECT_EQ(-2, c(0,1).as<int>());
    EXPECT_EQ(-4, c(0,2).as<int>());
    EXPECT_EQ(0, c(1,0).as<int>());
    EXPECT_EQ(8, c(1,1).as<int>());
    EXPECT_EQ(6, c(1,2).as<int>());

    a = 10;
    b = 20;

    c = nd::elwise(intfunc_other, a, b);
    EXPECT_EQ(-20, c.as<int>());

    a = aval0;
    b = bval0;
    c = nd::elwise(intfunc_other, a, b);
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

    c = nd::elwise(vecintfunc, a, b);
    EXPECT_EQ(20, c(0).as<int>());
    EXPECT_EQ(10, c(1).as<int>());

    int aval0[2][3] = {{0, 1, 2}, {5, 6, 7}};
    int bval0[3] = {5, 2, 4};
    a = aval0;
    b = bval0;
    c = nd::elwise(vecintfunc, a, b);

    EXPECT_EQ(ndt::type("strided * strided * 2 * int32"), c.get_type());
    ASSERT_EQ(2, c.get_shape()[0]);
    ASSERT_EQ(3, c.get_shape()[1]);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(bval0[j], c(i,j,0).as<int>());
            EXPECT_EQ(aval0[i][j], c(i,j,1).as<int>());
        }
    }

    c = nd::elwise(vecintfunc2, a, b, b);
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

    c = nd::elwise(IntMemFuncWrapper(), a, b);
    EXPECT_EQ(-20, c.as<int>());

    int aval0[2][3] = {{0, 1, 2}, {5, 6, 7}};
    int bval0[3] = {5, 2, 4};
    a = aval0;
    b = bval0;
    c = nd::elwise(IntMemFuncWrapper(), a, b);
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

void two_dim_res_one_arg_func(double (&res)[2][2], double x) {
    res[0][0] = cos(x);
    res[0][1] = -sin(x);
    res[1][0] = sin(x);
    res[1][1] = cos(x);
}

TEST(ArrayViews, FixedDimResOneArgFunc) {
    double x = 3.14 / 8;
    nd::array a = x, c;

    c = nd::elwise(two_dim_res_one_arg_func, a);
    EXPECT_EQ(ndt::type("2 * 2 * float64"), c.get_type());
    EXPECT_EQ(cos(x), c(0, 0).as<double>());
    EXPECT_EQ(-sin(x), c(0, 1).as<double>());
    EXPECT_EQ(sin(x), c(1, 0).as<double>());
    EXPECT_EQ(cos(x), c(1, 1).as<double>());
}

void one_dim_res_two_arg_func(int (&res)[1], const int x, int &y)
{
    res[0] = max(x, y);
}

void two_dim_res_two_arg_func(int (&res)[2], int x, int y)
{
    res[0] = y;
    res[1] = x;
}

TEST(ArrayViews, FixedDimResTwoArgFunc) {
    nd::array a = 10, b = 20, c;

    c = nd::elwise(one_dim_res_two_arg_func, a, b);
    EXPECT_EQ(20, c(0).as<int>());

    c = nd::elwise(two_dim_res_two_arg_func, a, b);
    EXPECT_EQ(20, c(0).as<int>());
    EXPECT_EQ(10, c(1).as<int>());

    int aval0[2][3] = {{0, 1, 2}, {5, 6, 7}};
    int bval0[3] = {5, 2, 4};
    a = aval0;
    b = bval0;
    c = nd::elwise(two_dim_res_two_arg_func, a, b);

    EXPECT_EQ(ndt::type("strided * strided * 2 * int32"), c.get_type());
    ASSERT_EQ(2, c.get_shape()[0]);
    ASSERT_EQ(3, c.get_shape()[1]);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(bval0[j], c(i,j,0).as<int>());
            EXPECT_EQ(aval0[i][j], c(i,j,1).as<int>());
        }
    }
}
