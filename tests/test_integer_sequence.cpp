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

using std::is_same;
using dynd::outer;
using dynd::integer_sequence;
using dynd::type_sequence;
using dynd::for_each;

template <typename T>
class IntegerSequence : public ::testing::Test {
};

TYPED_TEST_CASE_P(IntegerSequence);

template <typename T>
struct accumulator {
  template <T I>
  void on_each(T &res) const
  {
    res += I;
  }
};

TYPED_TEST_P(IntegerSequence, ForEach)
{
  typedef integer_sequence<TypeParam, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9> S;

  TypeParam res = 0;
  for_each<S>(accumulator<TypeParam>(), res);
  EXPECT_EQ(static_cast<TypeParam>(45), res);
}

TYPED_TEST_P(IntegerSequence, Outer)
{
  EXPECT_TRUE(
      (is_same<typename outer<integer_sequence<TypeParam, 0>,
                              integer_sequence<TypeParam, 1, 2>>::type,
               type_sequence<integer_sequence<TypeParam, 0, 1>,
                             integer_sequence<TypeParam, 0, 2>>>::value));

  EXPECT_TRUE(
      (is_same<typename outer<integer_sequence<TypeParam, 0, 1>,
                              integer_sequence<TypeParam, 2>>::type,
               type_sequence<integer_sequence<TypeParam, 0, 2>,
                             integer_sequence<TypeParam, 1, 2>>>::value));

  EXPECT_TRUE(
      (is_same<typename outer<integer_sequence<TypeParam, 0, 1>,
                              integer_sequence<TypeParam, 2, 3>>::type,
               type_sequence<integer_sequence<TypeParam, 0, 2>,
                             integer_sequence<TypeParam, 0, 3>,
                             integer_sequence<TypeParam, 1, 2>,
                             integer_sequence<TypeParam, 1, 3>>>::value));

  EXPECT_TRUE((is_same<
      typename outer<integer_sequence<TypeParam, 0, 1>,
                     integer_sequence<TypeParam, 2, 3, 4>>::type,
      type_sequence<
          integer_sequence<TypeParam, 0, 2>, integer_sequence<TypeParam, 0, 3>,
          integer_sequence<TypeParam, 0, 4>, integer_sequence<TypeParam, 1, 2>,
          integer_sequence<TypeParam, 1, 3>,
          integer_sequence<TypeParam, 1, 4>>>::value));

  EXPECT_TRUE((is_same<
      typename outer<integer_sequence<TypeParam, 0, 1, 2>,
                     integer_sequence<TypeParam, 3, 4>>::type,
      type_sequence<
          integer_sequence<TypeParam, 0, 3>, integer_sequence<TypeParam, 0, 4>,
          integer_sequence<TypeParam, 1, 3>, integer_sequence<TypeParam, 1, 4>,
          integer_sequence<TypeParam, 2, 3>,
          integer_sequence<TypeParam, 2, 4>>>::value));

  EXPECT_TRUE((is_same<
      typename outer<integer_sequence<TypeParam, 0, 1, 2, 3, 4>,
                     integer_sequence<TypeParam, 5, 6, 7, 8, 9>>::type,
      type_sequence<
          integer_sequence<TypeParam, 0, 5>, integer_sequence<TypeParam, 0, 6>,
          integer_sequence<TypeParam, 0, 7>, integer_sequence<TypeParam, 0, 8>,
          integer_sequence<TypeParam, 0, 9>, integer_sequence<TypeParam, 1, 5>,
          integer_sequence<TypeParam, 1, 6>, integer_sequence<TypeParam, 1, 7>,
          integer_sequence<TypeParam, 1, 8>, integer_sequence<TypeParam, 1, 9>,
          integer_sequence<TypeParam, 2, 5>, integer_sequence<TypeParam, 2, 6>,
          integer_sequence<TypeParam, 2, 7>, integer_sequence<TypeParam, 2, 8>,
          integer_sequence<TypeParam, 2, 9>, integer_sequence<TypeParam, 3, 5>,
          integer_sequence<TypeParam, 3, 6>, integer_sequence<TypeParam, 3, 7>,
          integer_sequence<TypeParam, 3, 8>, integer_sequence<TypeParam, 3, 9>,
          integer_sequence<TypeParam, 4, 5>, integer_sequence<TypeParam, 4, 6>,
          integer_sequence<TypeParam, 4, 7>, integer_sequence<TypeParam, 4, 8>,
          integer_sequence<TypeParam, 4, 9>>>::value));

  EXPECT_TRUE(
      (is_same<typename outer<integer_sequence<TypeParam, 0>,
                              integer_sequence<TypeParam, 1, 2>,
                              integer_sequence<TypeParam, 3, 4>>::type,
               type_sequence<integer_sequence<TypeParam, 0, 1, 3>,
                             integer_sequence<TypeParam, 0, 1, 4>,
                             integer_sequence<TypeParam, 0, 2, 3>,
                             integer_sequence<TypeParam, 0, 2, 4>>>::value));

  EXPECT_TRUE(
      (is_same<typename outer<integer_sequence<TypeParam, 0, 1>,
                              integer_sequence<TypeParam, 2>,
                              integer_sequence<TypeParam, 3, 4>>::type,
               type_sequence<integer_sequence<TypeParam, 0, 2, 3>,
                             integer_sequence<TypeParam, 0, 2, 4>,
                             integer_sequence<TypeParam, 1, 2, 3>,
                             integer_sequence<TypeParam, 1, 2, 4>>>::value));

  EXPECT_TRUE(
      (is_same<typename outer<integer_sequence<TypeParam, 0, 1>,
                              integer_sequence<TypeParam, 2, 3>,
                              integer_sequence<TypeParam, 4>>::type,
               type_sequence<integer_sequence<TypeParam, 0, 2, 4>,
                             integer_sequence<TypeParam, 0, 3, 4>,
                             integer_sequence<TypeParam, 1, 2, 4>,
                             integer_sequence<TypeParam, 1, 3, 4>>>::value));

  EXPECT_TRUE(
      (is_same<typename outer<integer_sequence<TypeParam, 0, 1>,
                              integer_sequence<TypeParam, 2, 3>,
                              integer_sequence<TypeParam, 4, 5>>::type,
               type_sequence<integer_sequence<TypeParam, 0, 2, 4>,
                             integer_sequence<TypeParam, 0, 2, 5>,
                             integer_sequence<TypeParam, 0, 3, 4>,
                             integer_sequence<TypeParam, 0, 3, 5>,
                             integer_sequence<TypeParam, 1, 2, 4>,
                             integer_sequence<TypeParam, 1, 2, 5>,
                             integer_sequence<TypeParam, 1, 3, 4>,
                             integer_sequence<TypeParam, 1, 3, 5>>>::value));

  EXPECT_TRUE(
      (is_same<typename outer<integer_sequence<TypeParam, 0, 1, 2>,
                              integer_sequence<TypeParam, 3, 4>,
                              integer_sequence<TypeParam, 5, 6>>::type,
               type_sequence<integer_sequence<TypeParam, 0, 3, 5>,
                             integer_sequence<TypeParam, 0, 3, 6>,
                             integer_sequence<TypeParam, 0, 4, 5>,
                             integer_sequence<TypeParam, 0, 4, 6>,
                             integer_sequence<TypeParam, 1, 3, 5>,
                             integer_sequence<TypeParam, 1, 3, 6>,
                             integer_sequence<TypeParam, 1, 4, 5>,
                             integer_sequence<TypeParam, 1, 4, 6>,
                             integer_sequence<TypeParam, 2, 3, 5>,
                             integer_sequence<TypeParam, 2, 3, 6>,
                             integer_sequence<TypeParam, 2, 4, 5>,
                             integer_sequence<TypeParam, 2, 4, 6>>>::value));

  EXPECT_TRUE(
      (is_same<typename outer<integer_sequence<TypeParam, 0, 1>,
                              integer_sequence<TypeParam, 2, 3, 4>,
                              integer_sequence<TypeParam, 5, 6>>::type,
               type_sequence<integer_sequence<TypeParam, 0, 2, 5>,
                             integer_sequence<TypeParam, 0, 2, 6>,
                             integer_sequence<TypeParam, 0, 3, 5>,
                             integer_sequence<TypeParam, 0, 3, 6>,
                             integer_sequence<TypeParam, 0, 4, 5>,
                             integer_sequence<TypeParam, 0, 4, 6>,
                             integer_sequence<TypeParam, 1, 2, 5>,
                             integer_sequence<TypeParam, 1, 2, 6>,
                             integer_sequence<TypeParam, 1, 3, 5>,
                             integer_sequence<TypeParam, 1, 3, 6>,
                             integer_sequence<TypeParam, 1, 4, 5>,
                             integer_sequence<TypeParam, 1, 4, 6>>>::value));

  EXPECT_TRUE(
      (is_same<typename outer<integer_sequence<TypeParam, 0, 1>,
                              integer_sequence<TypeParam, 2, 3>,
                              integer_sequence<TypeParam, 4, 5, 6>>::type,
               type_sequence<integer_sequence<TypeParam, 0, 2, 4>,
                             integer_sequence<TypeParam, 0, 2, 5>,
                             integer_sequence<TypeParam, 0, 2, 6>,
                             integer_sequence<TypeParam, 0, 3, 4>,
                             integer_sequence<TypeParam, 0, 3, 5>,
                             integer_sequence<TypeParam, 0, 3, 6>,
                             integer_sequence<TypeParam, 1, 2, 4>,
                             integer_sequence<TypeParam, 1, 2, 5>,
                             integer_sequence<TypeParam, 1, 2, 6>,
                             integer_sequence<TypeParam, 1, 3, 4>,
                             integer_sequence<TypeParam, 1, 3, 5>,
                             integer_sequence<TypeParam, 1, 3, 6>>>::value));

  EXPECT_TRUE(
      (is_same<typename outer<integer_sequence<TypeParam, 0, 1>,
                              integer_sequence<TypeParam, 2, 3>,
                              integer_sequence<TypeParam, 4, 5>,
                              integer_sequence<TypeParam, 6, 7>>::type,
               type_sequence<integer_sequence<TypeParam, 0, 2, 4, 6>,
                             integer_sequence<TypeParam, 0, 2, 4, 7>,
                             integer_sequence<TypeParam, 0, 2, 5, 6>,
                             integer_sequence<TypeParam, 0, 2, 5, 7>,
                             integer_sequence<TypeParam, 0, 3, 4, 6>,
                             integer_sequence<TypeParam, 0, 3, 4, 7>,
                             integer_sequence<TypeParam, 0, 3, 5, 6>,
                             integer_sequence<TypeParam, 0, 3, 5, 7>,
                             integer_sequence<TypeParam, 1, 2, 4, 6>,
                             integer_sequence<TypeParam, 1, 2, 4, 7>,
                             integer_sequence<TypeParam, 1, 2, 5, 6>,
                             integer_sequence<TypeParam, 1, 2, 5, 7>,
                             integer_sequence<TypeParam, 1, 3, 4, 6>,
                             integer_sequence<TypeParam, 1, 3, 4, 7>,
                             integer_sequence<TypeParam, 1, 3, 5, 6>,
                             integer_sequence<TypeParam, 1, 3, 5, 7>>>::value));

  EXPECT_TRUE((is_same<
      typename outer<
          integer_sequence<TypeParam, 0, 1>, integer_sequence<TypeParam, 2, 3>,
          integer_sequence<TypeParam, 4, 5>, integer_sequence<TypeParam, 6, 7>,
          integer_sequence<TypeParam, 8, 9>>::type,
      type_sequence<integer_sequence<TypeParam, 0, 2, 4, 6, 8>,
                    integer_sequence<TypeParam, 0, 2, 4, 6, 9>,
                    integer_sequence<TypeParam, 0, 2, 4, 7, 8>,
                    integer_sequence<TypeParam, 0, 2, 4, 7, 9>,
                    integer_sequence<TypeParam, 0, 2, 5, 6, 8>,
                    integer_sequence<TypeParam, 0, 2, 5, 6, 9>,
                    integer_sequence<TypeParam, 0, 2, 5, 7, 8>,
                    integer_sequence<TypeParam, 0, 2, 5, 7, 9>,
                    integer_sequence<TypeParam, 0, 3, 4, 6, 8>,
                    integer_sequence<TypeParam, 0, 3, 4, 6, 9>,
                    integer_sequence<TypeParam, 0, 3, 4, 7, 8>,
                    integer_sequence<TypeParam, 0, 3, 4, 7, 9>,
                    integer_sequence<TypeParam, 0, 3, 5, 6, 8>,
                    integer_sequence<TypeParam, 0, 3, 5, 6, 9>,
                    integer_sequence<TypeParam, 0, 3, 5, 7, 8>,
                    integer_sequence<TypeParam, 0, 3, 5, 7, 9>,
                    integer_sequence<TypeParam, 1, 2, 4, 6, 8>,
                    integer_sequence<TypeParam, 1, 2, 4, 6, 9>,
                    integer_sequence<TypeParam, 1, 2, 4, 7, 8>,
                    integer_sequence<TypeParam, 1, 2, 4, 7, 9>,
                    integer_sequence<TypeParam, 1, 2, 5, 6, 8>,
                    integer_sequence<TypeParam, 1, 2, 5, 6, 9>,
                    integer_sequence<TypeParam, 1, 2, 5, 7, 8>,
                    integer_sequence<TypeParam, 1, 2, 5, 7, 9>,
                    integer_sequence<TypeParam, 1, 3, 4, 6, 8>,
                    integer_sequence<TypeParam, 1, 3, 4, 6, 9>,
                    integer_sequence<TypeParam, 1, 3, 4, 7, 8>,
                    integer_sequence<TypeParam, 1, 3, 4, 7, 9>,
                    integer_sequence<TypeParam, 1, 3, 5, 6, 8>,
                    integer_sequence<TypeParam, 1, 3, 5, 6, 9>,
                    integer_sequence<TypeParam, 1, 3, 5, 7, 8>,
                    integer_sequence<TypeParam, 1, 3, 5, 7, 9>>>::value));
}

#ifndef _MSC_VER
REGISTER_TYPED_TEST_CASE_P(IntegerSequence, ForEach, Outer);
INSTANTIATE_TYPED_TEST_CASE_P(SizeType, IntegerSequence, size_t);
#endif