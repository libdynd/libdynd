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
  static T value;
};

template <typename T>
struct member_value_wrapper {
  T value;
};

DYND_HAS(value);

struct func_wrapper {
  static int func() { return 0; };
};

struct member_func_wrapper {
  int func() { return 0; };
};

DYND_HAS(func);

TEST(Config, Has)
{
  EXPECT_TRUE(has_value<value_wrapper<int>>::value);
  EXPECT_TRUE(has_value<value_wrapper<const char *>>::value);
  EXPECT_FALSE(has_value<::empty>::value);
  EXPECT_FALSE(has_value<member_value_wrapper<int>>::value);

  EXPECT_TRUE((has_value<value_wrapper<int>, int>::value));
  EXPECT_FALSE((has_value<value_wrapper<int>, const int>::value));
  EXPECT_FALSE((has_value<value_wrapper<int>, int &>::value));
  EXPECT_FALSE((has_value<value_wrapper<int>, const int &>::value));
  EXPECT_FALSE((has_value<value_wrapper<int>, bool>::value));
  EXPECT_FALSE((has_value<value_wrapper<bool>, int>::value));
  EXPECT_FALSE((has_value<::empty, int>::value));
  EXPECT_FALSE((has_value<member_value_wrapper<int>, int>::value));

  EXPECT_TRUE((has_value<value_wrapper<char *>, char *>::value));
  EXPECT_FALSE((has_value<value_wrapper<char *>, const char *>::value));
  EXPECT_FALSE((has_value<value_wrapper<char *>, int &>::value));
  EXPECT_FALSE((has_value<value_wrapper<char *>, const char *&>::value));
  EXPECT_FALSE((has_value<value_wrapper<char *>, bool>::value));
  EXPECT_FALSE((has_value<value_wrapper<bool>, char *>::value));
  EXPECT_FALSE((has_value<::empty, char *>::value));
  EXPECT_FALSE((has_value<member_value_wrapper<char *>, char *>::value));

  EXPECT_TRUE((has_func<func_wrapper, int()>::value));
  EXPECT_FALSE((has_func<func_wrapper, void()>::value));
  EXPECT_FALSE((has_func<func_wrapper, int>::value));
  EXPECT_FALSE((has_func<::empty, int()>::value));
  EXPECT_FALSE((has_func<member_func_wrapper, int()>::value));
}
