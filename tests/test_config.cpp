//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dynd_assertions.hpp"
#include "inc_gtest.hpp"

#include <dynd/config.hpp>

using namespace std;
using namespace dynd;

struct empty {};

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

TEST(Config, Has) {
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

DYND_HAS_MEMBER(func);

TEST(Config, HasMember) {
  EXPECT_TRUE((has_member_func<member_func_wrapper, int()>::value));
  EXPECT_FALSE((has_member_func<func_wrapper, int()>::value));
}

TEST(Config, Fold) {
  EXPECT_EQ(10, lfold<std::plus<int>>(0, 1, 2, 3, 4));
  EXPECT_EQ(24, lfold<std::multiplies<int>>(1, 2, 3, 4));
}

TEST(Config, Zip) {
  int i = 0;
  int j = 3;
  for (auto pair : zip({0, 1, 2}, {3, 4, 5})) {
    EXPECT_EQ(i, pair.first);
    EXPECT_EQ(j, pair.second);

    ++i;
    ++j;
  }
}

TEST(Config, Outer) {
  struct type0;
  struct type1;
  struct type2;
  struct type3;
  struct type4;
  struct type5;
  struct type6;
  struct type7;
  struct type8;
  struct type9;

  EXPECT_TRUE((is_same<outer_t<type_sequence<type0>, type_sequence<type1, type2>>,
                       type_sequence<type_sequence<type0, type1>, type_sequence<type0, type2>>>::value));

  EXPECT_TRUE((is_same<outer_t<type_sequence<type0, type1>, type_sequence<type2>>,
                       type_sequence<type_sequence<type0, type2>, type_sequence<type1, type2>>>::value));

  EXPECT_TRUE((is_same<outer_t<type_sequence<type0, type1>, type_sequence<type2, type3>>,
                       type_sequence<type_sequence<type0, type2>, type_sequence<type0, type3>,
                                     type_sequence<type1, type2>, type_sequence<type1, type3>>>::value));

  EXPECT_TRUE((is_same<outer_t<type_sequence<type0, type1>, type_sequence<type2, type3, type4>>,
                       type_sequence<type_sequence<type0, type2>, type_sequence<type0, type3>,
                                     type_sequence<type0, type4>, type_sequence<type1, type2>,
                                     type_sequence<type1, type3>, type_sequence<type1, type4>>>::value));

  EXPECT_TRUE((is_same<outer_t<type_sequence<type0, type1, type2>, type_sequence<type3, type4>>,
                       type_sequence<type_sequence<type0, type3>, type_sequence<type0, type4>,
                                     type_sequence<type1, type3>, type_sequence<type1, type4>,
                                     type_sequence<type2, type3>, type_sequence<type2, type4>>>::value));

  EXPECT_TRUE(
      (is_same<
          outer_t<type_sequence<type0, type1, type2, type3, type4>, type_sequence<type5, type6, type7, type8, type9>>,
          type_sequence<type_sequence<type0, type5>, type_sequence<type0, type6>, type_sequence<type0, type7>,
                        type_sequence<type0, type8>, type_sequence<type0, type9>, type_sequence<type1, type5>,
                        type_sequence<type1, type6>, type_sequence<type1, type7>, type_sequence<type1, type8>,
                        type_sequence<type1, type9>, type_sequence<type2, type5>, type_sequence<type2, type6>,
                        type_sequence<type2, type7>, type_sequence<type2, type8>, type_sequence<type2, type9>,
                        type_sequence<type3, type5>, type_sequence<type3, type6>, type_sequence<type3, type7>,
                        type_sequence<type3, type8>, type_sequence<type3, type9>, type_sequence<type4, type5>,
                        type_sequence<type4, type6>, type_sequence<type4, type7>, type_sequence<type4, type8>,
                        type_sequence<type4, type9>>>::value));

  EXPECT_TRUE((is_same<outer_t<type_sequence<type0>, type_sequence<type1, type2>, type_sequence<type3, type4>>,
                       type_sequence<type_sequence<type0, type1, type3>, type_sequence<type0, type1, type4>,
                                     type_sequence<type0, type2, type3>, type_sequence<type0, type2, type4>>>::value));

  EXPECT_TRUE((is_same<outer_t<type_sequence<type0, type1>, type_sequence<type2>, type_sequence<type3, type4>>,
                       type_sequence<type_sequence<type0, type2, type3>, type_sequence<type0, type2, type4>,
                                     type_sequence<type1, type2, type3>, type_sequence<type1, type2, type4>>>::value));

  EXPECT_TRUE((is_same<outer_t<type_sequence<type0, type1>, type_sequence<type2, type3>, type_sequence<type4>>,
                       type_sequence<type_sequence<type0, type2, type4>, type_sequence<type0, type3, type4>,
                                     type_sequence<type1, type2, type4>, type_sequence<type1, type3, type4>>>::value));

  EXPECT_TRUE((is_same<outer_t<type_sequence<type0, type1>, type_sequence<type2, type3>, type_sequence<type4, type5>>,
                       type_sequence<type_sequence<type0, type2, type4>, type_sequence<type0, type2, type5>,
                                     type_sequence<type0, type3, type4>, type_sequence<type0, type3, type5>,
                                     type_sequence<type1, type2, type4>, type_sequence<type1, type2, type5>,
                                     type_sequence<type1, type3, type4>, type_sequence<type1, type3, type5>>>::value));

  EXPECT_TRUE(
      (is_same<outer_t<type_sequence<type0, type1, type2>, type_sequence<type3, type4>, type_sequence<type5, type6>>,
               type_sequence<type_sequence<type0, type3, type5>, type_sequence<type0, type3, type6>,
                             type_sequence<type0, type4, type5>, type_sequence<type0, type4, type6>,
                             type_sequence<type1, type3, type5>, type_sequence<type1, type3, type6>,
                             type_sequence<type1, type4, type5>, type_sequence<type1, type4, type6>,
                             type_sequence<type2, type3, type5>, type_sequence<type2, type3, type6>,
                             type_sequence<type2, type4, type5>, type_sequence<type2, type4, type6>>>::value));

  EXPECT_TRUE(
      (is_same<outer_t<type_sequence<type0, type1>, type_sequence<type2, type3, type4>, type_sequence<type5, type6>>,
               type_sequence<type_sequence<type0, type2, type5>, type_sequence<type0, type2, type6>,
                             type_sequence<type0, type3, type5>, type_sequence<type0, type3, type6>,
                             type_sequence<type0, type4, type5>, type_sequence<type0, type4, type6>,
                             type_sequence<type1, type2, type5>, type_sequence<type1, type2, type6>,
                             type_sequence<type1, type3, type5>, type_sequence<type1, type3, type6>,
                             type_sequence<type1, type4, type5>, type_sequence<type1, type4, type6>>>::value));

  EXPECT_TRUE(
      (is_same<outer_t<type_sequence<type0, type1>, type_sequence<type2, type3>, type_sequence<type4, type5, type6>>,
               type_sequence<type_sequence<type0, type2, type4>, type_sequence<type0, type2, type5>,
                             type_sequence<type0, type2, type6>, type_sequence<type0, type3, type4>,
                             type_sequence<type0, type3, type5>, type_sequence<type0, type3, type6>,
                             type_sequence<type1, type2, type4>, type_sequence<type1, type2, type5>,
                             type_sequence<type1, type2, type6>, type_sequence<type1, type3, type4>,
                             type_sequence<type1, type3, type5>, type_sequence<type1, type3, type6>>>::value));

  EXPECT_TRUE(
      (is_same<
          outer_t<type_sequence<type0, type1>, type_sequence<type2, type3>, type_sequence<type4, type5>,
                  type_sequence<type6, type7>>,
          type_sequence<type_sequence<type0, type2, type4, type6>, type_sequence<type0, type2, type4, type7>,
                        type_sequence<type0, type2, type5, type6>, type_sequence<type0, type2, type5, type7>,
                        type_sequence<type0, type3, type4, type6>, type_sequence<type0, type3, type4, type7>,
                        type_sequence<type0, type3, type5, type6>, type_sequence<type0, type3, type5, type7>,
                        type_sequence<type1, type2, type4, type6>, type_sequence<type1, type2, type4, type7>,
                        type_sequence<type1, type2, type5, type6>, type_sequence<type1, type2, type5, type7>,
                        type_sequence<type1, type3, type4, type6>, type_sequence<type1, type3, type4, type7>,
                        type_sequence<type1, type3, type5, type6>, type_sequence<type1, type3, type5, type7>>>::value));

  EXPECT_TRUE(
      (is_same<outer_t<type_sequence<type0, type1>, type_sequence<type2, type3>, type_sequence<type4, type5>,
                       type_sequence<type6, type7>, type_sequence<type8, type9>>,
               type_sequence<
                   type_sequence<type0, type2, type4, type6, type8>, type_sequence<type0, type2, type4, type6, type9>,
                   type_sequence<type0, type2, type4, type7, type8>, type_sequence<type0, type2, type4, type7, type9>,
                   type_sequence<type0, type2, type5, type6, type8>, type_sequence<type0, type2, type5, type6, type9>,
                   type_sequence<type0, type2, type5, type7, type8>, type_sequence<type0, type2, type5, type7, type9>,
                   type_sequence<type0, type3, type4, type6, type8>, type_sequence<type0, type3, type4, type6, type9>,
                   type_sequence<type0, type3, type4, type7, type8>, type_sequence<type0, type3, type4, type7, type9>,
                   type_sequence<type0, type3, type5, type6, type8>, type_sequence<type0, type3, type5, type6, type9>,
                   type_sequence<type0, type3, type5, type7, type8>, type_sequence<type0, type3, type5, type7, type9>,
                   type_sequence<type1, type2, type4, type6, type8>, type_sequence<type1, type2, type4, type6, type9>,
                   type_sequence<type1, type2, type4, type7, type8>, type_sequence<type1, type2, type4, type7, type9>,
                   type_sequence<type1, type2, type5, type6, type8>, type_sequence<type1, type2, type5, type6, type9>,
                   type_sequence<type1, type2, type5, type7, type8>, type_sequence<type1, type2, type5, type7, type9>,
                   type_sequence<type1, type3, type4, type6, type8>, type_sequence<type1, type3, type4, type6, type9>,
                   type_sequence<type1, type3, type4, type7, type8>, type_sequence<type1, type3, type4, type7, type9>,
                   type_sequence<type1, type3, type5, type6, type8>, type_sequence<type1, type3, type5, type6, type9>,
                   type_sequence<type1, type3, type5, type7, type8>,
                   type_sequence<type1, type3, type5, type7, type9>>>::value));
}
