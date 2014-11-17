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
class ElwiseMethRetRes : public ::testing::Test {
};

TYPED_TEST_CASE_P(ElwiseMethRetRes);

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
class FuncWrapper;

template <typename R, typename A0>
class FuncWrapper<R (*)(A0)> {
private:
    R (*m_func)(A0);
public:
    FuncWrapper(R (*func)(A0)) : m_func(func) {}
    R meth(A0 a0) const {
        return (*m_func)(a0);
    }
};
template <typename R, typename A0, typename A1>
class FuncWrapper<R (*)(A0, A1)> {
private:
    R (*m_func)(A0, A1);
public:
    FuncWrapper(R (*func)(A0, A1)) : m_func(func) {}
    R meth(A0 a0, A1 a1) const {
        return (*m_func)(a0, a1);
    }
};

TYPED_TEST_P(ElwiseMethRetRes, MethRetRes) {
    typedef FuncWrapper<int (*)(TypeParam, const TypeParam &)> FuncWrapper0;
    typedef FuncWrapper<TypeParam (*)(const TypeParam (&)[3])> FuncWrapper1;
    typedef FuncWrapper<TypeParam (*)(const TypeParam (&)[3], const TypeParam (&)[3])> FuncWrapper2;
    typedef FuncWrapper<TypeParam (*)(const TypeParam (&)[2][3])> FuncWrapper3;

    nd::array res, a, b;

    a = static_cast<TypeParam>(10);
    b = static_cast<TypeParam>(20);

    res = nd::elwise(FuncWrapper0(&func0), &FuncWrapper0::meth, a, b);
    EXPECT_EQ(-20, res.as<int>());

    TypeParam avals0[2][3] = {{0, 1, 2}, {5, 6, 7}};
    TypeParam bvals0[3] = {5, 2, 4};

    a = avals0;
    b = bvals0;
    res = nd::elwise(FuncWrapper0(&func0), &FuncWrapper0::meth, a, b);
    EXPECT_EQ(ndt::type("2 * 3 * int"), res.get_type());
    EXPECT_JSON_EQ_ARR("[[-10,-2,-4], [0,8,6]]", res);

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
}

typedef ::testing::Types<int, float, long, double> test_types;

REGISTER_TYPED_TEST_CASE_P(ElwiseMethRetRes, MethRetRes);
INSTANTIATE_TYPED_TEST_CASE_P(Builtin, ElwiseMethRetRes, test_types);
*/
