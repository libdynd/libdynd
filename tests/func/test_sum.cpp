//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/arithmetic.hpp>
#include <dynd/logic.hpp>
#include <dynd_assertions.hpp>

using namespace std;
using namespace dynd;

/*
TEST(Sum, 1D)
{
  // int32
  EXPECT_ARRAY_EQ(1, nd::sum(nd::array{1}));
  EXPECT_ARRAY_EQ(-1, nd::sum(nd::array{1, -2}));
  EXPECT_ARRAY_EQ(11, nd::sum(nd::array{1, -2, 12}));

  // int64
  EXPECT_ARRAY_EQ(1LL, nd::sum(nd::array{1LL}));
  EXPECT_ARRAY_EQ(-19999999999LL, nd::sum(nd::array{1LL, -20000000000LL}));
  EXPECT_ARRAY_EQ(-19999999987LL, nd::sum(nd::array{1LL, -20000000000LL, 12LL}));

  // float32
  EXPECT_ARRAY_EQ(1.25f, nd::sum(nd::array{1.25f}));
  EXPECT_ARRAY_EQ(-1.25f, nd::sum(nd::array{1.25f, -2.5f}));
  EXPECT_ARRAY_EQ(10.875f, nd::sum(nd::array{1.25f, -2.5f, 12.125f}));

  // float64
  EXPECT_ARRAY_EQ(1.25, nd::sum(nd::array{1.25}));
  EXPECT_ARRAY_EQ(-1.25, nd::sum(nd::array{1.25, -2.5}));
  EXPECT_ARRAY_EQ(10.875, nd::sum(nd::array{1.25, -2.5, 12.125}));

  // complex[float32]
  EXPECT_ARRAY_EQ(dynd::complex<float>(1.25f, -2.125f), nd::sum(nd::array{dynd::complex<float>(1.25f, -2.125f)}));
  EXPECT_ARRAY_EQ(dynd::complex<float>(-1.25f, -1.125f),
                  nd::sum(nd::array{dynd::complex<float>(1.25f, -2.125f), dynd::complex<float>(-2.5f, 1.0f)}));
  EXPECT_ARRAY_EQ(dynd::complex<float>(10.875f, 12343.875f),
                  nd::sum(nd::array{dynd::complex<float>(1.25f, -2.125f), dynd::complex<float>(-2.5f, 1.0f),
                                    dynd::complex<float>(12.125f, 12345.0f)}));

  // complex[float64]
  EXPECT_ARRAY_EQ(dynd::complex<double>(1.25, -2.125), nd::sum(nd::array{dynd::complex<double>(1.25, -2.125)}));
  EXPECT_ARRAY_EQ(dynd::complex<double>(-1.25, -1.125),
                  nd::sum(nd::array{dynd::complex<double>(1.25, -2.125), dynd::complex<double>(-2.5, 1.0)}));
  EXPECT_ARRAY_EQ(dynd::complex<double>(10.875, 12343.875),
                  nd::sum(nd::array{dynd::complex<double>(1.25, -2.125), dynd::complex<double>(-2.5, 1.0),
                                    dynd::complex<double>(12.125, 12345.0)}));
}
*/

/*
TEST(Sum, 2D)
{
  EXPECT_ARRAY_EQ(15, nd::sum(nd::array{{0, 1, 2}, {3, 4, 5}}));
}
*/
