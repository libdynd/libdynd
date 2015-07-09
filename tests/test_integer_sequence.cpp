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

template <typename T>
struct pop_front {
  typedef typename from<T, 1>::type type;
};

template <typename... S>
struct outer;

template <size_t I0, size_t... I1>
struct outer<index_sequence<I0>, index_sequence<I1...>> {
  typedef type_sequence<index_sequence<I0, I1>...> type;
};

template <size_t... I0, size_t... I1>
struct outer<type_sequence<index_sequence<I0...>>, index_sequence<I1...>> {
  typedef type_sequence<index_sequence<I0..., I1>...> type;
};

template <typename S0, typename S1>
struct outer<S0, S1> {
  typedef typename join<
      typename outer<typename to<S0, 1>::type, S1>::type,
      typename outer<typename pop_front<S0>::type, S1>::type>::type type;
};

template <typename S0, typename S1, typename... S>
struct outer<S0, S1, S...> {
  typedef typename outer<typename outer<S0, S1>::type, S...>::type type;
};

} // namespace dynd

TEST(IndexSequence, Outer)
{
  EXPECT_TRUE((std::is_same<
      typename outer<index_sequence<0>, index_sequence<1, 2>>::type,
      type_sequence<index_sequence<0, 1>, index_sequence<0, 2>>>::value));

  EXPECT_TRUE((std::is_same<
      typename outer<index_sequence<0, 1>, index_sequence<2>>::type,
      type_sequence<index_sequence<0, 2>, index_sequence<1, 2>>>::value));

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

  EXPECT_TRUE((std::is_same<
      typename outer<index_sequence<0>, index_sequence<1, 2>,
                     index_sequence<3, 4>>::type,
      type_sequence<index_sequence<0, 1, 3>, index_sequence<0, 1, 4>,
                    index_sequence<0, 2, 3>, index_sequence<0, 2, 4>>>::value));

  EXPECT_TRUE((std::is_same<
      typename outer<index_sequence<0, 1>, index_sequence<2>,
                     index_sequence<3, 4>>::type,
      type_sequence<index_sequence<0, 2, 3>, index_sequence<0, 2, 4>,
                    index_sequence<1, 2, 3>, index_sequence<1, 2, 4>>>::value));

  EXPECT_TRUE((std::is_same<
      typename outer<index_sequence<0, 1>, index_sequence<2, 3>,
                     index_sequence<4>>::type,
      type_sequence<index_sequence<0, 2, 4>, index_sequence<0, 3, 4>,
                    index_sequence<1, 2, 4>, index_sequence<1, 3, 4>>>::value));

  EXPECT_TRUE((std::is_same<
      typename outer<index_sequence<0, 1>, index_sequence<2, 3>,
                     index_sequence<4, 5>>::type,
      type_sequence<index_sequence<0, 2, 4>, index_sequence<0, 2, 5>,
                    index_sequence<0, 3, 4>, index_sequence<0, 3, 5>,
                    index_sequence<1, 2, 4>, index_sequence<1, 2, 5>,
                    index_sequence<1, 3, 4>, index_sequence<1, 3, 5>>>::value));

  EXPECT_TRUE((std::is_same<
      typename outer<index_sequence<0, 1, 2>, index_sequence<3, 4>,
                     index_sequence<5, 6>>::type,
      type_sequence<index_sequence<0, 3, 5>, index_sequence<0, 3, 6>,
                    index_sequence<0, 4, 5>, index_sequence<0, 4, 6>,
                    index_sequence<1, 3, 5>, index_sequence<1, 3, 6>,
                    index_sequence<1, 4, 5>, index_sequence<1, 4, 6>,
                    index_sequence<2, 3, 5>, index_sequence<2, 3, 6>,
                    index_sequence<2, 4, 5>, index_sequence<2, 4, 6>>>::value));

  EXPECT_TRUE((std::is_same<
      typename outer<index_sequence<0, 1>, index_sequence<2, 3, 4>,
                     index_sequence<5, 6>>::type,
      type_sequence<index_sequence<0, 2, 5>, index_sequence<0, 2, 6>,
                    index_sequence<0, 3, 5>, index_sequence<0, 3, 6>,
                    index_sequence<0, 4, 5>, index_sequence<0, 4, 6>,
                    index_sequence<1, 2, 5>, index_sequence<1, 2, 6>,
                    index_sequence<1, 3, 5>, index_sequence<1, 3, 6>,
                    index_sequence<1, 4, 5>, index_sequence<1, 4, 6>>>::value));

  EXPECT_TRUE((std::is_same<
      typename outer<index_sequence<0, 1>, index_sequence<2, 3>,
                     index_sequence<4, 5, 6>>::type,
      type_sequence<index_sequence<0, 2, 4>, index_sequence<0, 2, 5>,
                    index_sequence<0, 2, 6>, index_sequence<0, 3, 4>,
                    index_sequence<0, 3, 5>, index_sequence<0, 3, 6>,
                    index_sequence<1, 2, 4>, index_sequence<1, 2, 5>,
                    index_sequence<1, 2, 6>, index_sequence<1, 3, 4>,
                    index_sequence<1, 3, 5>, index_sequence<1, 3, 6>>>::value));

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
}