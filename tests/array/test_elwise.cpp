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

void func_ref_res_0(int &res, int x, int y) {
    res = 2 * (x - y);
}

void func_ref_res_1(int &res, int (&x)[3]) {
    res = x[0] + x[1] + x[2];
}

void func_ref_res_2(int &res, int (&x)[3], int (&y)[3]) {
    res = x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

void func_ref_res_3(int &res, int(&x)[2][3]) {
    res = x[0][0] + x[0][1] + x[1][2];
}

void func_ref_res_4(int (&res)[2], int x, int y) {
    res[0] = y;
    res[1] = x;
}

void func_ref_res_5(int (&res)[3], int(&x)[3][3], int(&y)[3]) {
    res[0] = x[0][0] * y[0] + x[0][1] * y[1] + x[0][2] * y[2];
    res[1] = x[1][0] * y[0] + x[1][1] * y[1] + x[1][2] * y[2];
    res[2] = x[2][0] * y[0] + x[2][1] * y[1] + x[2][2] * y[2];
}

void func_ref_res_6(double (&res)[2][2], int x) {
    res[0][0] = cos((double) x);
    res[0][1] = -sin((double) x);
    res[1][0] = sin((double) x);
    res[1][1] = cos((double) x);
}

void func_ref_res_7(int (&res)[3][3], int(&x)[3][3], int(&y)[3][3]) {
    res[0][0] = x[0][0] * y[0][0] + x[0][1] * y[1][0] + x[0][2] * y[2][0];
    res[0][1] = x[0][0] * y[0][1] + x[0][1] * y[1][1] + x[0][2] * y[2][1];
    res[0][2] = x[0][0] * y[0][2] + x[0][1] * y[1][2] + x[0][2] * y[2][2];
    res[1][0] = x[1][0] * y[0][0] + x[1][1] * y[1][0] + x[1][2] * y[2][0];
    res[1][1] = x[1][0] * y[0][1] + x[1][1] * y[1][1] + x[1][2] * y[2][1];
    res[1][2] = x[1][0] * y[0][2] + x[1][1] * y[1][2] + x[1][2] * y[2][2];
    res[2][0] = x[2][0] * y[0][0] + x[2][1] * y[1][0] + x[2][2] * y[2][0];
    res[2][1] = x[2][0] * y[0][1] + x[2][1] * y[1][1] + x[2][2] * y[2][1];
    res[2][2] = x[2][0] * y[0][2] + x[2][1] * y[1][2] + x[2][2] * y[2][2];
}

TEST(ArrayViews, IntFuncRefRes) {
    nd::array res, a, b;

    a = 10;
    b = 20;

    res = nd::elwise(func_ref_res_0, a, b);
    EXPECT_EQ(-20, res.as<int>());

    res = nd::elwise(func_ref_res_4, a, b);
    EXPECT_EQ(ndt::type("2 * int32"), res.get_type());
    EXPECT_EQ(20, res(0).as<int>());
    EXPECT_EQ(10, res(1).as<int>());

    a = 1;

    res = nd::elwise(func_ref_res_6, a);
    EXPECT_EQ(ndt::type("2 * 2 * float64"), res.get_type());
    EXPECT_EQ(cos((double) 1), res(0,0).as<double>());
    EXPECT_EQ(-sin((double) 1), res(0,1).as<double>());
    EXPECT_EQ(sin((double) 1), res(1,0).as<double>());
    EXPECT_EQ(cos((double) 1), res(1,1).as<double>());

    int aval_0[2][3] = {{0, 1, 2}, {5, 6, 7}};
    int bval_0[3] = {5, 2, 4};

    a = aval_0;
    b = bval_0;
    res = nd::elwise(func_ref_res_0, a, b);
    EXPECT_EQ(ndt::type("strided * strided * int32"), res.get_type());
    ASSERT_EQ(2, res.get_shape()[0]);
    ASSERT_EQ(3, res.get_shape()[1]);
    EXPECT_EQ(-10, res(0, 0).as<int>());
    EXPECT_EQ(-2, res(0, 1).as<int>());
    EXPECT_EQ(-4, res(0, 2).as<int>());
    EXPECT_EQ(0, res(1, 0).as<int>());
    EXPECT_EQ(8, res(1, 1).as<int>());
    EXPECT_EQ(6, res(1, 2).as<int>());

    res = nd::elwise(func_ref_res_4, a, b);
    EXPECT_EQ(ndt::type("strided * strided * 2 * int32"), res.get_type());
    ASSERT_EQ(2, res.get_shape()[0]);
    ASSERT_EQ(3, res.get_shape()[1]);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(bval_0[j], res(i, j, 0).as<int>());
            EXPECT_EQ(aval_0[i][j], res(i, j, 1).as<int>());
        }
    }

    int vals_1[2][3] = {{0, 1, 2}, {3, 4, 5}};

    a = nd::empty(ndt::make_fixed_dim(3, ndt::make_type<int>()));

    a.vals() = vals_1[0];
    res = nd::elwise(func_ref_res_1, a);
    EXPECT_EQ(ndt::type("int32"), res.get_type());
    EXPECT_EQ(3, res.as<int>());

    a.vals() = vals_1[1];
    res = nd::elwise(func_ref_res_1, a);
    EXPECT_EQ(ndt::type("int32"), res.get_type());
    EXPECT_EQ(12, res.as<int>());

    b = nd::empty(ndt::make_fixed_dim(3, ndt::make_type<int>()));

    a.vals() = vals_1[0];
    b.vals() = vals_1[1];
    res = nd::elwise(func_ref_res_2, a, b);
    EXPECT_EQ(ndt::type("int32"), res.get_type());
    EXPECT_EQ(14, res.as<int>());

    a = nd::empty(ndt::fixed_dim_from_array<int[2][3]>::make());

    a.vals() = vals_1;
    res = nd::elwise(func_ref_res_3, a);
    EXPECT_EQ(ndt::type("int32"), res.get_type());
    EXPECT_EQ(6, res.as<int>());

    int avals_2[3][3] = {{8, -7, 6}, {5, -4, 3}, {2, -1, 0}};
    int bvals_2[3] = {33, 7, 53401};

    a = nd::empty(ndt::fixed_dim_from_array<int[3][3]>::make());

    a.vals() = avals_2;
    b.vals() = bvals_2;
    res = nd::elwise(func_ref_res_5, a, b);
    EXPECT_EQ(ndt::type("3 * int32"), res.get_type());
    for (int i = 0; i < 3; ++i) {
            EXPECT_EQ(res(i), avals_2[i][0] * bvals_2[0] + avals_2[i][1] * bvals_2[1]
                + avals_2[i][2] * bvals_2[2]);
    }

    int bvals_3[3][3] = {{12, 0, -5}, {33, 7, 53401}, {64, 952, 8}};

    b = nd::empty(a.get_type());

    b.vals() = bvals_3;
    res = nd::elwise(func_ref_res_7, a, b);
    EXPECT_EQ(ndt::type("3 * 3 * int32"), res.get_type());
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(res(i, j), avals_2[i][0] * bvals_3[0][j] + avals_2[i][1] * bvals_3[1][j]
                + avals_2[i][2] * bvals_3[2][j]);
        }
    }
}

/*
    Everything below here is outdated.
*/

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
