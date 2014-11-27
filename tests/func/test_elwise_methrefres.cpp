//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/types/cfixed_dim_type.hpp>

/*
using namespace std;
using namespace dynd;

template <typename T>
class ElwiseMethRefRes : public ::testing::Test {
};

TYPED_TEST_CASE_P(ElwiseMethRefRes);

template <typename T>
int func0(T x, const T &y) {
    return static_cast<int>(2 * (x - y));
}
template <typename T>
T func1(const T (&x)[3]) {
    return x[0] + x[1] + x[2];
}
template <typename T>
T func2(const T (&x)[3], const T (&y)[3]) {
    return static_cast<T>(x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
}
template <typename T>
T func3(const T (&x)[2][3]) {
    return x[0][0] + x[0][1] + x[1][2];
}
template <typename T>
void func0(int &res, T x, const T &y) {
    res = func0(x, y);
}
template <typename T>
void func1(T &res, const T (&x)[3]) {
    res = func1(x);
}
template <typename T>
void func2(T &res, const T (&x)[3], const T (&y)[3]) {
    res = func2(x, y);
}
template <typename T>
void func3(T &res, const T (&x)[2][3]) {
    res = func3(x);
}
template <typename T>
void func4(T (&res)[2], T x, T y) {
    res[0] = y;
    res[1] = x;
}
template <typename T>
void func5(T (&res)[3], const T(&x)[3][3], const T(&y)[3]) {
    res[0] = x[0][0] * y[0] + x[0][1] * y[1] + x[0][2] * y[2];
    res[1] = x[1][0] * y[0] + x[1][1] * y[1] + x[1][2] * y[2];
    res[2] = x[2][0] * y[0] + x[2][1] * y[1] + x[2][2] * y[2];
}
template <typename T>
void func6(double (&res)[2][2], T x) {
    res[0][0] = cos((double) x);
    res[0][1] = -sin((double) x);
    res[1][0] = sin((double) x);
    res[1][1] = cos((double) x);
}
template <typename T>
void func7(T (&res)[3][3], const T(&x)[3][3], const T(&y)[3][3]) {
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

template <typename T>
class FuncWrapper;

template <typename R, typename A0>
class FuncWrapper<void (*)(R &, A0)> {
private:
    void (*m_func)(R &, A0);
public:
    FuncWrapper(void (*func)(R &, A0)) : m_func(func) {}
    void meth(R &res, A0 a0) const {
        (*m_func)(res, a0);
    }
};
template <typename R, typename A0, typename A1>
class FuncWrapper<void (*)(R &, A0, A1)> {
private:
    void (*m_func)(R &, A0, A1);
public:
    FuncWrapper(void (*func)(R &, A0, A1)) : m_func(func) {}
    void meth(R &res, A0 a0, A1 a1) const {
        (*m_func)(res, a0, a1);
    }
};

TYPED_TEST_P(ElwiseMethRefRes, MethRefRes) {
    typedef FuncWrapper<void (*)(int &, TypeParam, const TypeParam &)> FuncWrapper0;
    typedef FuncWrapper<void (*)(TypeParam &, const TypeParam (&)[3])> FuncWrapper1;
    typedef FuncWrapper<void (*)(TypeParam &, const TypeParam (&)[3], const TypeParam (&)[3])> FuncWrapper2;
    typedef FuncWrapper<void (*)(TypeParam &, const TypeParam (&)[2][3])> FuncWrapper3;
    typedef FuncWrapper<void (*)(TypeParam (&)[2], TypeParam, TypeParam)> FuncWrapper4;
    typedef FuncWrapper<void (*)(TypeParam (&)[3], const TypeParam(&)[3][3], const TypeParam(&)[3])> FuncWrapper5;
    typedef FuncWrapper<void (*)(double (&)[2][2], TypeParam)> FuncWrapper6;
    typedef FuncWrapper<void (*)(TypeParam (&)[3][3], const TypeParam(&)[3][3], const TypeParam(&)[3][3])> FuncWrapper7;

    nd::array res, a, b;

    a = static_cast<TypeParam>(10);
    b = static_cast<TypeParam>(20);

    res = nd::elwise(FuncWrapper0(&func0), &FuncWrapper0::meth, a, b);
    EXPECT_EQ(-20, res.as<int>());

    res = nd::elwise(FuncWrapper4(&func4), &FuncWrapper4::meth, a, b);
    EXPECT_EQ(ndt::make_cfixed_dim(2, ndt::make_type<TypeParam>()), res.get_type());
    EXPECT_EQ(20, res(0).as<TypeParam>());
    EXPECT_EQ(10, res(1).as<TypeParam>());

    a = static_cast<TypeParam>(1);

    res = nd::elwise(FuncWrapper6(&func6), &FuncWrapper6::meth, a);
    EXPECT_EQ(ndt::type("cfixed[2] * cfixed[2] * float64"), res.get_type());
    EXPECT_EQ(cos((double) 1), res(0,0).as<double>());
    EXPECT_EQ(-sin((double) 1), res(0,1).as<double>());
    EXPECT_EQ(sin((double) 1), res(1,0).as<double>());
    EXPECT_EQ(cos((double) 1), res(1,1).as<double>());

    TypeParam avals0[2][3] = {{0, 1, 2}, {5, 6, 7}};
    TypeParam bvals0[3] = {5, 2, 4};

    a = avals0;
    b = bvals0;
    res = nd::elwise(FuncWrapper0(&func0), &FuncWrapper0::meth, a, b);
    EXPECT_EQ(ndt::type("2 * 3 * int"), res.get_type());
    EXPECT_JSON_EQ_ARR("[[-10,-2,-4], [0,8,6]]", res);

    res = nd::elwise(FuncWrapper4(&func4), &FuncWrapper4::meth, a, b);
    EXPECT_EQ(
        ndt::make_fixed_dim(
            2, ndt::make_fixed_dim(
                   3, ndt::make_cfixed_dim(2, ndt::make_type<TypeParam>()))),
        res.get_type());
    ASSERT_EQ(2, res.get_shape()[0]);
    ASSERT_EQ(3, res.get_shape()[1]);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(bvals0[j], res(i, j, 0).as<TypeParam>());
            EXPECT_EQ(avals0[i][j], res(i, j, 1).as<TypeParam>());
        }
    }

    TypeParam vals1[2][3] = {{0, 1, 2}, {3, 4, 5}};

    a = nd::empty(ndt::make_cfixed_dim(3, ndt::make_type<TypeParam>()));

    a.vals() = vals1[0];
    res = nd::elwise(FuncWrapper1(&func1), &FuncWrapper1::meth, a);
    EXPECT_EQ(ndt::make_type<TypeParam>(), res.get_type());
    EXPECT_EQ(3, res.as<TypeParam>());

    a.vals() = vals1[1];
    res = nd::elwise(FuncWrapper1(&func1), &FuncWrapper1::meth, a);
    EXPECT_EQ(ndt::make_type<TypeParam>(), res.get_type());
    EXPECT_EQ(12, res.as<TypeParam>());

    b = nd::empty(ndt::make_cfixed_dim(3, ndt::make_type<TypeParam>()));

    a.vals() = vals1[0];
    b.vals() = vals1[1];
    res = nd::elwise(FuncWrapper2(&func2), &FuncWrapper2::meth, a, b);
    EXPECT_EQ(ndt::make_type<TypeParam>(), res.get_type());
    EXPECT_EQ(14, res.as<TypeParam>());

    a = nd::empty(ndt::cfixed_dim_from_array<TypeParam[2][3]>::make());

    a.vals() = vals1;
    res = nd::elwise(FuncWrapper3(&func3), &FuncWrapper3::meth, a);
    EXPECT_EQ(ndt::make_type<TypeParam>(), res.get_type());
    EXPECT_EQ(6, res.as<TypeParam>());

    TypeParam avals2[3][3] = {{8, -7, (TypeParam) 6.353},
        {(TypeParam) 5.432423, -4, 3},
        {2, (TypeParam) -1.7023, 0}};
    TypeParam bvals2[3] = {33, 7, 53401};

    a = nd::empty(ndt::cfixed_dim_from_array<TypeParam[3][3]>::make());

    a.vals() = avals2;
    b.vals() = bvals2;
    res = nd::elwise(FuncWrapper5(&func5), &FuncWrapper5::meth, a, b);
    EXPECT_EQ(ndt::make_cfixed_dim(3, ndt::make_type<TypeParam>()), res.get_type());
    for (int i = 0; i < 3; ++i) {
            EXPECT_EQ(res(i), avals2[i][0] * bvals2[0] + avals2[i][1] * bvals2[1]
                + avals2[i][2] * bvals2[2]);
    }

    TypeParam bvals_3[3][3] = {{(TypeParam) 12.4, 0, -5},
        {(TypeParam) 33.5, (TypeParam) 7.2, 53401},
        {(TypeParam) 64.512, 952, (TypeParam) 8.1}};

    b = nd::empty(a.get_type());

    b.vals() = bvals_3;
    res = nd::elwise(FuncWrapper7(&func7), &FuncWrapper7::meth, a, b);
    EXPECT_EQ(ndt::make_cfixed_dim(3, ndt::make_cfixed_dim(3, ndt::make_type<TypeParam>())), res.get_type());
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(res(i, j), avals2[i][0] * bvals_3[0][j] + avals2[i][1] * bvals_3[1][j]
                + avals2[i][2] * bvals_3[2][j]);
        }
    }
}

typedef ::testing::Types<int, float, long, double> test_types;

REGISTER_TYPED_TEST_CASE_P(ElwiseMethRefRes, MethRefRes);
INSTANTIATE_TYPED_TEST_CASE_P(Builtin, ElwiseMethRefRes, test_types);
*/
