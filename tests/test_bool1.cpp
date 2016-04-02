//
// Copyright (C) 2011-16 DyND Developers
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

//#define EXPECT_TYPE_IS_SAME(A, B) EXPECT_TRUE((std::is_same<A, B>::value))

TEST(Bool1, ArithmeticType)
{
// int32 + uint32 -> uint32

//  EXPECT_TYPE_IS_SAME(decltype(declval<bool1>() / declval<short>()),
//                    (typename common_type<bool, short>::type));

#define EXPECTATIONS(OPERATOR)                                                                                         \
  EXPECT_TRUE((                                                                                                        \
      is_same<decltype(declval<bool1>() OPERATOR declval<short>()), typename common_type<bool, short>::type>::value)); \
  EXPECT_TRUE(                                                                                                         \
      (is_same<decltype(declval<bool1>() OPERATOR declval<int>()), typename common_type<bool, int>::type>::value));    \
  EXPECT_TRUE(                                                                                                         \
      (is_same<decltype(declval<bool1>() OPERATOR declval<long>()), typename common_type<bool, long>::type>::value));  \
  EXPECT_TRUE((is_same<decltype(declval<bool1>() OPERATOR declval<long long>()),                                       \
                       typename common_type<bool, long long>::type>::value));                                          \
  EXPECT_TRUE((is_same<decltype(declval<bool1>() OPERATOR declval<unsigned char>()),                                   \
                       typename common_type<bool, unsigned char>::type>::value));                                      \
  EXPECT_TRUE((is_same<decltype(declval<bool1>() OPERATOR declval<unsigned short>()),                                  \
                       typename common_type<bool, unsigned short>::type>::value));                                     \
  EXPECT_TRUE((is_same<decltype(declval<bool1>() OPERATOR declval<unsigned int>()),                                    \
                       typename common_type<bool, unsigned int>::type>::value));                                       \
  EXPECT_TRUE((is_same<decltype(declval<bool1>() OPERATOR declval<unsigned long>()),                                   \
                       typename common_type<bool, unsigned long>::type>::value));                                      \
  EXPECT_TRUE((is_same<decltype(declval<bool1>() OPERATOR declval<unsigned long long>()),                              \
                       typename common_type<bool, unsigned long long>::type>::value));                                 \
  EXPECT_TRUE((                                                                                                        \
      is_same<decltype(declval<bool1>() OPERATOR declval<float>()), typename common_type<bool, float>::type>::value)); \
  EXPECT_TRUE((is_same<decltype(declval<bool1>() OPERATOR declval<double>()),                                          \
                       typename common_type<bool, double>::type>::value));                                             \
                                                                                                                       \
  EXPECT_TRUE((is_same<decltype(declval<signed char>() OPERATOR declval<bool1>()),                                     \
                       typename common_type<signed char, bool>::type>::value));                                        \
  EXPECT_TRUE((                                                                                                        \
      is_same<decltype(declval<short>() OPERATOR declval<bool1>()), typename common_type<short, bool>::type>::value)); \
  EXPECT_TRUE(                                                                                                         \
      (is_same<decltype(declval<int>() OPERATOR declval<bool1>()), typename common_type<int, bool>::type>::value));    \
  EXPECT_TRUE(                                                                                                         \
      (is_same<decltype(declval<long>() OPERATOR declval<bool1>()), typename common_type<long, bool1>::type>::value)); \
  EXPECT_TRUE((is_same<decltype(declval<long long>() OPERATOR declval<bool1>()),                                       \
                       typename common_type<long long, bool>::type>::value));                                          \
  EXPECT_TRUE((is_same<decltype(declval<unsigned char>() OPERATOR declval<bool1>()),                                   \
                       typename common_type<unsigned char, bool>::type>::value));                                      \
  EXPECT_TRUE((is_same<decltype(declval<bool1>() OPERATOR declval<unsigned short>()),                                  \
                       typename common_type<bool, unsigned short>::type>::value));                                     \
  EXPECT_TRUE((is_same<decltype(declval<bool1>() OPERATOR declval<unsigned int>()),                                    \
                       typename common_type<bool, unsigned int>::type>::value));                                       \
  EXPECT_TRUE((is_same<decltype(declval<bool1>() OPERATOR declval<unsigned long>()),                                   \
                       typename common_type<bool, unsigned long>::type>::value));                                      \
  EXPECT_TRUE((is_same<decltype(declval<bool1>() OPERATOR declval<unsigned long long>()),                              \
                       typename common_type<bool, unsigned long long>::type>::value));                                 \
  EXPECT_TRUE((                                                                                                        \
      is_same<decltype(declval<bool1>() OPERATOR declval<float>()), typename common_type<bool, float>::type>::value)); \
  EXPECT_TRUE((is_same<decltype(declval<bool1>() OPERATOR declval<double>()),                                          \
                       typename common_type<bool, double>::type>::value))

  EXPECTATIONS(/ );
}

TEST(Complex, CommonType)
{
  using dynd::complex;
  typedef double T;

  EXPECT_TRUE((is_same<decltype(declval<bool1>() / declval<complex<float>>()), complex<float>>::value));
  EXPECT_TRUE((is_same<decltype(declval<bool1>() / declval<complex<double>>()), complex<double>>::value));

  EXPECT_TRUE((is_same<decltype(declval<complex<T>>() / declval<signed char>()),
                       complex<typename common_type<T, signed char>::type>>::value));
  EXPECT_TRUE((is_same<decltype(declval<complex<T>>() / declval<short>()),
                       complex<typename common_type<T, short>::type>>::value));
  EXPECT_TRUE(
      (is_same<decltype(declval<complex<T>>() / declval<int>()), complex<typename common_type<T, int>::type>>::value));
  EXPECT_TRUE((
      is_same<decltype(declval<complex<T>>() / declval<long>()), complex<typename common_type<T, long>::type>>::value));
  EXPECT_TRUE((is_same<decltype(declval<complex<T>>() / declval<long long>()),
                       complex<typename common_type<T, long long>::type>>::value));
  EXPECT_TRUE((is_same<decltype(declval<complex<T>>() / declval<unsigned char>()),
                       complex<typename common_type<T, unsigned char>::type>>::value));
  EXPECT_TRUE((is_same<decltype(declval<complex<T>>() / declval<unsigned short>()),
                       complex<typename common_type<T, unsigned short>::type>>::value));
  EXPECT_TRUE((is_same<decltype(declval<complex<T>>() / declval<unsigned int>()),
                       complex<typename common_type<T, unsigned int>::type>>::value));
  EXPECT_TRUE((is_same<decltype(declval<complex<T>>() / declval<unsigned long>()),
                       complex<typename common_type<T, unsigned long>::type>>::value));
  EXPECT_TRUE((is_same<decltype(declval<complex<T>>() / declval<unsigned long long>()),
                       complex<typename common_type<T, unsigned long long>::type>>::value));
  EXPECT_TRUE((is_same<decltype(declval<complex<T>>() / declval<float>()),
                       complex<typename common_type<T, float>::type>>::value));
  EXPECT_TRUE((is_same<decltype(declval<complex<T>>() / declval<double>()),
                       complex<typename common_type<T, double>::type>>::value));

  EXPECT_TRUE((is_same<decltype(declval<complex<T>>() / declval<int128>()), complex<T>>::value));
}

TEST(Int128, CommonType)
{
  using dynd::complex;

  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<bool1>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<signed char>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<short>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<int>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<long>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<long long>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<bool1>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<unsigned char>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<unsigned short>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<unsigned int>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<unsigned long>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<unsigned long long>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<float>()), float>::value));
  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<double>()), double>::value));
  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<complex<float>>()), complex<float>>::value));
  EXPECT_TRUE((is_same<decltype(declval<int128>() / declval<complex<double>>()), complex<double>>::value));

  EXPECT_TRUE((is_same<decltype(declval<bool1>() / declval<int128>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<signed char>() / declval<int128>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<short>() / declval<int128>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<int>() / declval<int128>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<long>() / declval<int128>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<long long>() / declval<int128>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<unsigned char>() / declval<int128>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<unsigned short>() / declval<int128>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<unsigned int>() / declval<int128>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<unsigned long>() / declval<int128>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<unsigned long long>() / declval<int128>()), int128>::value));
  EXPECT_TRUE((is_same<decltype(declval<float>() / declval<int128>()), float>::value));
  EXPECT_TRUE((is_same<decltype(declval<double>() / declval<int128>()), double>::value));

  /*
    std::cout
        << ndt::type::make<decltype(declval<int>() + declval<unsigned
    short>())>()
        << std::endl;
    std::exit(-1);
  */
}
