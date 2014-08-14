//
// Copyright (C) 2011-14 Irwin Zaid, Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/func/functor_arrfunc.hpp>

using namespace std;
using namespace dynd;

template <typename T>
class FunctorArrfunc_FuncRetRes : public ::testing::Test {
};

TYPED_TEST_CASE_P(FunctorArrfunc_FuncRetRes);

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

TYPED_TEST_P(FunctorArrfunc_FuncRetRes, FuncRetRes) {
  nd::array res, a, b;

  a = static_cast<TypeParam>(10);
  b = static_cast<TypeParam>(20);

  nd::arrfunc af = nd::make_functor_arrfunc(
      static_cast<int (*)(TypeParam, const TypeParam &)>(&func0));
  res = af(a, b);
  EXPECT_EQ(-20, res.as<int>());

  TypeParam vals1[2][3] = {{0, 1, 2}, {3, 4, 5}};

  a = nd::empty(ndt::make_cfixed_dim(3, ndt::make_type<TypeParam>()));

  a.vals() = vals1[0];
  af = nd::make_functor_arrfunc(
      static_cast<TypeParam (*)(const TypeParam(&)[3])>(&func1));
  res = af(a);
  EXPECT_EQ(ndt::make_type<TypeParam>(), res.get_type());
  EXPECT_EQ(3, res.as<TypeParam>());

  a.vals() = vals1[1];
  af = nd::make_functor_arrfunc(
      static_cast<TypeParam (*)(const TypeParam(&)[3])>(&func1));
  res = af(a);
  EXPECT_EQ(ndt::make_type<TypeParam>(), res.get_type());
  EXPECT_EQ(12, res.as<TypeParam>());

  b = nd::empty(ndt::make_cfixed_dim(3, ndt::make_type<TypeParam>()));

  a.vals() = vals1[0];
  b.vals() = vals1[1];
  af = nd::make_functor_arrfunc(
      static_cast<TypeParam (*)(const TypeParam(&)[3], const TypeParam(&)[3])>(
          &func2));
  res = af(a, b);
  EXPECT_EQ(ndt::make_type<TypeParam>(), res.get_type());
  EXPECT_EQ(14, res.as<TypeParam>());

  a = nd::empty(ndt::cfixed_dim_from_array<TypeParam[2][3]>::make());

  a.vals() = vals1;
  af = nd::make_functor_arrfunc(
      static_cast<TypeParam (*)(const TypeParam(&)[2][3])>(&func3));
  res = af(a);
  EXPECT_EQ(ndt::make_type<TypeParam>(), res.get_type());
  EXPECT_EQ(6, res.as<TypeParam>());
}

typedef ::testing::Types<int, float, long, double> types;

REGISTER_TYPED_TEST_CASE_P(FunctorArrfunc_FuncRetRes, FuncRetRes);
INSTANTIATE_TYPED_TEST_CASE_P(Builtin, FunctorArrfunc_FuncRetRes, types);
