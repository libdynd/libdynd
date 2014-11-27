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
class ElwiseCallRetRes : public ::testing::Test {
};

TYPED_TEST_CASE_P(ElwiseCallRetRes);

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
class Callable;

template <typename R, typename A0>
class Callable<R (*)(A0)> {
private:
    R (*m_func)(A0);
public:
    Callable(R (*func)(A0)) : m_func(func) {}

    R operator ()(A0 a0) const {
        return (*m_func)(a0);
    }
};
template <typename R, typename A0, typename A1>
class Callable<R (*)(A0, A1)> {
private:
    R (*m_func)(A0, A1);
public:
    Callable(R (*func)(A0, A1)) : m_func(func) {
    }

    R operator ()(A0 a0, A1 a1) const {
        return (*m_func)(a0, a1);
    }
};

TYPED_TEST_P(ElwiseCallRetRes, CallRetRes) {
    typedef Callable<int (*)(TypeParam, const TypeParam &)> Callable0;
    typedef Callable<TypeParam (*)(const TypeParam (&)[3])> Callable1;
    typedef Callable<TypeParam (*)(const TypeParam (&)[3], const TypeParam (&)[3])> Callable2;
    typedef Callable<TypeParam (*)(const TypeParam (&)[2][3])> Callable3;

    nd::array res, a, b;

    a = static_cast<TypeParam>(10);
    b = static_cast<TypeParam>(20);

    res = nd::elwise(Callable0(&func0), a, b);
    EXPECT_EQ(-20, res.as<int>());

    TypeParam avals0[2][3] = {{0, 1, 2}, {5, 6, 7}};
    TypeParam bvals0[3] = {5, 2, 4};

    a = avals0;
    b = bvals0;
    res = nd::elwise(Callable0(&func0), a, b);
    EXPECT_EQ(ndt::type("2 * 3 * int"), res.get_type());
    EXPECT_JSON_EQ_ARR("[[-10,-2,-4], [0,8,6]]", res);

    TypeParam vals1[2][3] = {{0, 1, 2}, {3, 4, 5}};

    a = nd::empty(ndt::make_cfixed_dim(3, ndt::make_type<TypeParam>()));

    a.vals() = vals1[0];
    res = nd::elwise(Callable1(&func1), a);
    EXPECT_EQ(ndt::make_type<TypeParam>(), res.get_type());
    EXPECT_EQ(3, res.as<TypeParam>());

    a.vals() = vals1[1];
    res = nd::elwise(Callable1(&func1), a);
    EXPECT_EQ(ndt::make_type<TypeParam>(), res.get_type());
    EXPECT_EQ(12, res.as<TypeParam>());

    b = nd::empty(ndt::make_cfixed_dim(3, ndt::make_type<TypeParam>()));

    a.vals() = vals1[0];
    b.vals() = vals1[1];
    res = nd::elwise(Callable2(&func2), a, b);
    EXPECT_EQ(ndt::make_type<TypeParam>(), res.get_type());
    EXPECT_EQ(14, res.as<TypeParam>());

    a = nd::empty(ndt::cfixed_dim_from_array<TypeParam[2][3]>::make());

    a.vals() = vals1;
    res = nd::elwise(Callable3(&func3), a);
    EXPECT_EQ(ndt::make_type<TypeParam>(), res.get_type());
    EXPECT_EQ(6, res.as<TypeParam>());
}

#if defined(DYND_CXX_LAMBDAS) && !defined(__clang__)
TEST(ElWise, LambdaFunc) {
    nd::array a, b, res;

    a = 100;
    b = 1.5;
    res = nd::elwise([](int x, double y) {
        return (float)(x + y);
    }, a, b);
    EXPECT_EQ(ndt::make_type<float>(), res.get_type());
    EXPECT_EQ(101.5f, res.as<float>());

    double a_val[3] = {1.5, 2.0, 3.125};
    a = a_val;
    b = 3.25;
    res = nd::elwise([](double x, double y, int z) {
        return x * z + y;
    }, a, b, 10);
    EXPECT_EQ(ndt::type("3 * float64"), res.get_type());
    EXPECT_JSON_EQ_ARR("[18.25,23.25,34.5]", res);
}
#endif

typedef ::testing::Types<int, float, long, double> test_types;

REGISTER_TYPED_TEST_CASE_P(ElwiseCallRetRes, CallRetRes);
INSTANTIATE_TYPED_TEST_CASE_P(Builtin, ElwiseCallRetRes, test_types);
*/