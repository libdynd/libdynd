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

namespace dynd {

/*
template <typename I, size_t J>
struct append;

template <size_t... I, size_t J>
struct append<index_sequence<I...>, J> {
  typedef index_sequence<I..., J> type;
};
*/

template <typename... I>
struct outer;

template <size_t I0, size_t... J>
struct outer<index_sequence<I0>, index_sequence<J...>> {
  typedef type_sequence<index_sequence<I0, J>...> type;
};

template <typename I, typename... J>
struct outer<I, J...> {
  typedef typename join<
      typename outer<index_sequence<at<I, 0>::value>, J...>::type,
      typename outer<typename from<I, 1>::type, J...>::type>::type type;
};

/*
template <size_t I0, size_t... I, typename... J>
struct outer<index_sequence<I0, I...>, J...> {
  typedef typename join<typename outer<index_sequence<I0>, J...>::type,
                        typename outer<index_sequence<I...>, J...>::type>::type
      type;
};
*/

/*
template <typename I, size_t... J>
struct outer<type_sequence<I>, index_sequence<J...>> {
  typedef type_sequence<typename append<I, J>::type...> type;
};

template <typename I0, typename... I, typename J>
struct outer<type_sequence<I0, I...>, J> {
  typedef typename join<typename outer<type_sequence<I0>, J>::type,
                        typename outer<type_sequence<I...>, J>::type>::type
      type;
};

template <typename I0, typename I1, typename... I>
struct outer<I0, I1, I...> {
  typedef typename outer<typename outer<I0, I1>::type, I...>::type type;
};
*/

} // namespace dynd

TEST(IndexSequence, Outer)
{
  EXPECT_TRUE((std::is_same<
      typename outer<index_sequence<0>, index_sequence<1, 2>>::type,
      type_sequence<index_sequence<0, 1>, index_sequence<0, 2>>>::value));

  EXPECT_TRUE((std::is_same<
      typename outer<index_sequence<0, 1>, index_sequence<2>>::type,
      type_sequence<index_sequence<0, 2>, index_sequence<1, 2>>>::value));

  /*
    EXPECT_TRUE((std::is_same<
        typename outer<index_sequence<0, 1>, index_sequence<2, 3>>::type,
        type_sequence<index_sequence<0, 2>, index_sequence<0, 3>,
                      index_sequence<1, 2>, index_sequence<1, 3>>>::value));

    EXPECT_TRUE((std::is_same<
        typename outer<index_sequence<0, 1>, index_sequence<2, 3, 4>>::type,
        type_sequence<index_sequence<0, 2>, index_sequence<0, 3>,
                      index_sequence<0, 4>, index_sequence<1, 2>,
                      index_sequence<1, 3>, index_sequence<1, 4>>>::value));

    EXPECT_TRUE((std::is_same<
        typename outer<index_sequence<0, 1, 2>, index_sequence<3, 4>>::type,
        type_sequence<index_sequence<0, 3>, index_sequence<0, 4>,
                      index_sequence<1, 3>, index_sequence<1, 4>,
                      index_sequence<2, 3>, index_sequence<2, 4>>>::value));

    EXPECT_TRUE((std::is_same<
        typename outer<index_sequence<0, 1, 2, 3, 4>,
                       index_sequence<5, 6, 7, 8, 9>>::type,
        type_sequence<
            index_sequence<0, 5>, index_sequence<0, 6>, index_sequence<0, 7>,
            index_sequence<0, 8>, index_sequence<0, 9>, index_sequence<1, 5>,
            index_sequence<1, 6>, index_sequence<1, 7>, index_sequence<1, 8>,
            index_sequence<1, 9>, index_sequence<2, 5>, index_sequence<2, 6>,
            index_sequence<2, 7>, index_sequence<2, 8>, index_sequence<2, 9>,
            index_sequence<3, 5>, index_sequence<3, 6>, index_sequence<3, 7>,
            index_sequence<3, 8>, index_sequence<3, 9>, index_sequence<4, 5>,
            index_sequence<4, 6>, index_sequence<4, 7>, index_sequence<4, 8>,
            index_sequence<4, 9>>>::value));
  */

  /*
    EXPECT_TRUE((std::is_same<
        typename outer<index_sequence<0>, index_sequence<1, 2>,
                       index_sequence<3, 4>>::type,
        type_sequence<index_sequence<0, 1, 3>, index_sequence<0, 1, 4>,
                      index_sequence<0, 2, 3>, index_sequence<0, 2,
    4>>>::value));

    EXPECT_TRUE((std::is_same<
        typename outer<index_sequence<0, 1>, index_sequence<2>,
                       index_sequence<3, 4>>::type,
        type_sequence<index_sequence<0, 2, 3>, index_sequence<0, 2, 4>,
                      index_sequence<1, 2, 3>, index_sequence<1, 2,
    4>>>::value));

    EXPECT_TRUE((std::is_same<
        typename outer<index_sequence<0, 1>, index_sequence<2, 3>,
                       index_sequence<4>>::type,
        type_sequence<index_sequence<0, 2, 4>, index_sequence<0, 3, 4>,
                      index_sequence<1, 2, 4>, index_sequence<1, 3,
    4>>>::value));

    EXPECT_TRUE((std::is_same<
        typename outer<index_sequence<0, 1>, index_sequence<2, 3>,
                       index_sequence<4, 5>>::type,
        type_sequence<index_sequence<0, 2, 4>, index_sequence<0, 2, 5>,
                      index_sequence<0, 3, 4>, index_sequence<0, 3, 5>,
                      index_sequence<1, 2, 4>, index_sequence<1, 2, 5>,
                      index_sequence<1, 3, 4>, index_sequence<1, 3,
    5>>>::value));

    EXPECT_TRUE((std::is_same<
        typename outer<index_sequence<0, 1, 2>, index_sequence<3, 4>,
                       index_sequence<5, 6>>::type,
        type_sequence<index_sequence<0, 3, 5>, index_sequence<0, 3, 6>,
                      index_sequence<0, 4, 5>, index_sequence<0, 4, 6>,
                      index_sequence<1, 3, 5>, index_sequence<1, 3, 6>,
                      index_sequence<1, 4, 5>, index_sequence<1, 4, 6>,
                      index_sequence<2, 3, 5>, index_sequence<2, 3, 6>,
                      index_sequence<2, 4, 5>, index_sequence<2, 4,
    6>>>::value));

    EXPECT_TRUE((std::is_same<
        typename outer<index_sequence<0, 1>, index_sequence<2, 3, 4>,
                       index_sequence<5, 6>>::type,
        type_sequence<index_sequence<0, 2, 5>, index_sequence<0, 2, 6>,
                      index_sequence<0, 3, 5>, index_sequence<0, 3, 6>,
                      index_sequence<0, 4, 5>, index_sequence<0, 4, 6>,
                      index_sequence<1, 2, 5>, index_sequence<1, 2, 6>,
                      index_sequence<1, 3, 5>, index_sequence<1, 3, 6>,
                      index_sequence<1, 4, 5>, index_sequence<1, 4,
    6>>>::value));

    EXPECT_TRUE((std::is_same<
        typename outer<index_sequence<0, 1>, index_sequence<2, 3>,
                       index_sequence<4, 5, 6>>::type,
        type_sequence<index_sequence<0, 2, 4>, index_sequence<0, 2, 5>,
                      index_sequence<0, 2, 6>, index_sequence<0, 3, 4>,
                      index_sequence<0, 3, 5>, index_sequence<0, 3, 6>,
                      index_sequence<1, 2, 4>, index_sequence<1, 2, 5>,
                      index_sequence<1, 2, 6>, index_sequence<1, 3, 4>,
                      index_sequence<1, 3, 5>, index_sequence<1, 3,
    6>>>::value));

    EXPECT_TRUE((std::is_same<
        typename outer<index_sequence<0, 1>, index_sequence<2, 3>,
                       index_sequence<4, 5>, index_sequence<6, 7>>::type,
        type_sequence<index_sequence<0, 2, 4, 6>, index_sequence<0, 2, 4, 7>,
                      index_sequence<0, 2, 5, 6>, index_sequence<0, 2, 5, 7>,
                      index_sequence<0, 3, 4, 6>, index_sequence<0, 3, 4, 7>,
                      index_sequence<0, 3, 5, 6>, index_sequence<0, 3, 5, 7>,
                      index_sequence<1, 2, 4, 6>, index_sequence<1, 2, 4, 7>,
                      index_sequence<1, 2, 5, 6>, index_sequence<1, 2, 5, 7>,
                      index_sequence<1, 3, 4, 6>, index_sequence<1, 3, 4, 7>,
                      index_sequence<1, 3, 5, 6>,
                      index_sequence<1, 3, 5, 7>>>::value));

    EXPECT_TRUE((std::is_same<
        typename outer<index_sequence<0, 1>, index_sequence<2, 3>,
                       index_sequence<4, 5>, index_sequence<6, 7>,
                       index_sequence<8, 9>>::type,
        type_sequence<
            index_sequence<0, 2, 4, 6, 8>, index_sequence<0, 2, 4, 6, 9>,
            index_sequence<0, 2, 4, 7, 8>, index_sequence<0, 2, 4, 7, 9>,
            index_sequence<0, 2, 5, 6, 8>, index_sequence<0, 2, 5, 6, 9>,
            index_sequence<0, 2, 5, 7, 8>, index_sequence<0, 2, 5, 7, 9>,
            index_sequence<0, 3, 4, 6, 8>, index_sequence<0, 3, 4, 6, 9>,
            index_sequence<0, 3, 4, 7, 8>, index_sequence<0, 3, 4, 7, 9>,
            index_sequence<0, 3, 5, 6, 8>, index_sequence<0, 3, 5, 6, 9>,
            index_sequence<0, 3, 5, 7, 8>, index_sequence<0, 3, 5, 7, 9>,
            index_sequence<1, 2, 4, 6, 8>, index_sequence<1, 2, 4, 6, 9>,
            index_sequence<1, 2, 4, 7, 8>, index_sequence<1, 2, 4, 7, 9>,
            index_sequence<1, 2, 5, 6, 8>, index_sequence<1, 2, 5, 6, 9>,
            index_sequence<1, 2, 5, 7, 8>, index_sequence<1, 2, 5, 7, 9>,
            index_sequence<1, 3, 4, 6, 8>, index_sequence<1, 3, 4, 6, 9>,
            index_sequence<1, 3, 4, 7, 8>, index_sequence<1, 3, 4, 7, 9>,
            index_sequence<1, 3, 5, 6, 8>, index_sequence<1, 3, 5, 6, 9>,
            index_sequence<1, 3, 5, 7, 8>,
            index_sequence<1, 3, 5, 7, 9>>>::value));
  */
}