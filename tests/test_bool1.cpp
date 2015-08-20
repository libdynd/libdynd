//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/config.hpp>

#include <typeinfo>

using namespace std;
using namespace dynd;

/*
template <typename T>
class Bool1 : public testing::Test {
};

TYPED_TEST_CASE_P(Bool1);

typedef ::testing::Types<char, short, int, long, unsigned char, unsigned short,
                         unsigned int, unsigned long> IntegralTypes;
typedef ::testing::Types<float, double, long double> FloatingPointTypes;

TYPED_TEST_P(Bool1, Arithmetic)
{
  EXPECT_TRUE(
      (is_same<decltype(declval<bool>() / declval<TypeParam>()),
               decltype(declval<bool1>() / declval<TypeParam>())>::value));
  EXPECT_TRUE(
      (is_same<decltype(declval<bool1>() / declval<TypeParam>()),
               decltype(declval<bool>() / declval<TypeParam>())>::value));
}

REGISTER_TYPED_TEST_CASE_P(Bool1, Arithmetic);

INSTANTIATE_TYPED_TEST_CASE_P(Integral, Bool1, IntegralTypes);
INSTANTIATE_TYPED_TEST_CASE_P(FloatingPoint, Bool1, FloatingPointTypes);
*/
