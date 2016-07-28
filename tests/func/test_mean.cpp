//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/statistics.hpp>
#include <dynd_assertions.hpp>

using namespace std;
using namespace dynd;

/*
ToDo: Fix this.

TEST(Mean, 1D)
{
  EXPECT_ARRAY_EQ(0.0, nd::mean(nd::array{0.0}));
  EXPECT_ARRAY_EQ(1.0, nd::mean(nd::array{1.0}));
  EXPECT_ARRAY_EQ(2.0, nd::mean(nd::array{0.0, 2.0, 4.0}));
  EXPECT_ARRAY_EQ(3.0, nd::mean(nd::array{1.0, 3.0, 5.0}));
  EXPECT_ARRAY_EQ(4.5, nd::mean(nd::array{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}));
}

TEST(Mean, 2D)
{
  EXPECT_ARRAY_EQ(4.5, nd::mean(nd::array({{0.0, 1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0, 9.0}})));
  EXPECT_ARRAY_EQ(4.5, nd::mean(nd::array({{9.0, 8.0, 7.0, 6.0, 5.0}, {4.0, 3.0, 2.0, 1.0, 0.0}})));
}

*/
