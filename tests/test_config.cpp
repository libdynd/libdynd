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

using namespace std;
using namespace dynd;

struct empty {
};

template <typename T>
struct value_wrapper {
  T value;
};

template <typename T>
struct static_value_wrapper {
  static T value;
};

DYND_HAS_STATIC_MEMBER(value);

TEST(Config, HasStaticMember)
{
  EXPECT_TRUE(has_static_value<static_value_wrapper<int>>::value);
  EXPECT_TRUE(has_static_value<static_value_wrapper<const char *>>::value);
  EXPECT_FALSE(has_static_value<empty>::value);

  EXPECT_TRUE((has_static_value<static_value_wrapper<int>, int>::value));
  EXPECT_FALSE((has_static_value<static_value_wrapper<int>, const int>::value));
  EXPECT_FALSE((has_static_value<static_value_wrapper<int>, int &>::value));
  EXPECT_FALSE(
      (has_static_value<static_value_wrapper<int>, const int &>::value));
  EXPECT_FALSE((has_static_value<static_value_wrapper<int>, bool>::value));
  EXPECT_FALSE((has_static_value<static_value_wrapper<bool>, int>::value));
  EXPECT_FALSE((has_static_value<empty, int>::value));

  EXPECT_TRUE((has_static_value<static_value_wrapper<char *>, char *>::value));
  EXPECT_FALSE(
      (has_static_value<static_value_wrapper<char *>, const char *>::value));
  EXPECT_FALSE((has_static_value<static_value_wrapper<char *>, int &>::value));
  EXPECT_FALSE(
      (has_static_value<static_value_wrapper<char *>, const char *&>::value));
  EXPECT_FALSE((has_static_value<static_value_wrapper<char *>, bool>::value));
  EXPECT_FALSE((has_static_value<static_value_wrapper<bool>, char *>::value));
  EXPECT_FALSE((has_static_value<empty, char *>::value));
}