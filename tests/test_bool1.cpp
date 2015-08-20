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

template <class T>
class Bool1 : public testing::Test {
};

typedef ::testing::Types<int> Implementations;
TYPED_TEST_CASE(Bool1, Implementations);

TYPED_TEST(Bool1, DefaultConstructor) {

}

/*
template <typename T, typename U>
struct is_std_equivalent {
  static const bool value = std::is_same<T, U>::value;
};

template <>
struct is_std_equivalent<bool, bool1> {
  static const bool value = true;
};

template <>
struct is_std_equivalent<bool1, bool> {
  static const bool value = true;
};

TYPED_TEST_P(Bool1, Arithmetic)
{
}
*/
