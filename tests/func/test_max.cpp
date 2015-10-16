//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/func/min.hpp>
#include <dynd/func/max.hpp>

#include "dynd_assertions.hpp"

using namespace std;
using namespace dynd;

TEST(Min, FixedDim)
{
  EXPECT_ARRAY_EQ(0, nd::min(nd::array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  EXPECT_ARRAY_EQ(-9, nd::min(nd::array{0, -1, -2, -3, -4, -5, -6, -7, -8, -9}));
  EXPECT_ARRAY_EQ(-51, nd::min(nd::array{0, -1, -2, -3, 4, -51, -6, -7, -8, -9}));
  EXPECT_ARRAY_EQ(0.0, nd::min(nd::array{0.5, 1.5, 0.0}));
}

TEST(Min, FixedDimFixedDim)
{
  EXPECT_ARRAY_EQ(0, nd::min(nd::array{{0, 1}, {2, 3}, {4, 5}}));
  EXPECT_ARRAY_EQ(-2, nd::min(nd::array{{0, 1}, {-2, 9}, {4, 5}}));
  EXPECT_ARRAY_EQ(-0.5, nd::min(nd::array{{0.0, 1.0}, {-0.5, 7.5}, {4.0, 5.0}}));
}

TEST(Min, FixedDimVarDim)
{
  EXPECT_ARRAY_EQ(0, nd::min(parse_json(ndt::type("2 * var * int32"), "[[0], [1, 2]]")));
  EXPECT_ARRAY_EQ(-10, nd::min(parse_json(ndt::type("2 * var * int32"), "[[2], [-10, 0]]")));
  EXPECT_ARRAY_EQ(-4.0, nd::min(parse_json(ndt::type("3 * var * float64"), "[[23.5], [10, 2, 15], [-4]]")));
}

TEST(Max, FixedDim)
{
  EXPECT_ARRAY_EQ(9, nd::max(nd::array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  EXPECT_ARRAY_EQ(0, nd::max(nd::array{0, -1, -2, -3, -4, -5, -6, -7, -8, -9}));
  EXPECT_ARRAY_EQ(4, nd::max(nd::array{0, -1, -2, -3, 4, -5, -6, -7, -8, -9}));
  EXPECT_ARRAY_EQ(1.5, nd::max(nd::array{0.5, 1.5, 0.75}));
}

TEST(Max, FixedDimFixedDim)
{
  EXPECT_ARRAY_EQ(5, nd::max(nd::array{{0, 1}, {2, 3}, {4, 5}}));
  EXPECT_ARRAY_EQ(9, nd::max(nd::array{{0, 1}, {2, 9}, {4, 5}}));
  EXPECT_ARRAY_EQ(7.5, nd::max(nd::array{{0.0, 1.0}, {2.0, 7.5}, {4.0, 5.0}}));
}

TEST(Max, FixedDimVarDim)
{
  EXPECT_ARRAY_EQ(2, nd::max(parse_json(ndt::type("2 * var * int32"), "[[0], [1, 2]]")));
  EXPECT_ARRAY_EQ(10, nd::max(parse_json(ndt::type("2 * var * int32"), "[[0], [10, 2]]")));
  EXPECT_ARRAY_EQ(23.5, nd::max(parse_json(ndt::type("3 * var * float64"), "[[23.5], [10, 2, 15], [-4]]")));
}
