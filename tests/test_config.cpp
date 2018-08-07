//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/config.hpp>
#include <dynd/gtest.hpp>

using namespace std;
using namespace dynd;

TEST(Config, Fold) {
  EXPECT_EQ(10, lfold<std::plus<int>>(0, 1, 2, 3, 4));
  EXPECT_EQ(24, lfold<std::multiplies<int>>(1, 2, 3, 4));
}

TEST(Config, Zip) {
  array<int, 3> x{0, 1, 2};
  array<int, 3> y{3, 4, 5};

  int i = 0;
  int j = 3;
  for (auto pair : zip(x, y)) {
    EXPECT_EQ(i, pair.first);
    EXPECT_EQ(j, pair.second);

    ++i;
    ++j;
  }
}

/*
TEST(Config, Zip2) {
  int i = 0;
  int j = 3;
  for (auto pair : zip({0, 1, 2}, {3, 4, 5})) {
    EXPECT_EQ(i, pair.first);
    EXPECT_EQ(j, pair.second);

    ++i;
    ++j;
  }
}
*/

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
