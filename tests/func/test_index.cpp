//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/index.hpp>

#include "dynd_assertions.hpp"

using namespace std;
using namespace dynd;

TEST(Index, Dim)
{
  nd::array a = {0, 1, 2, 3, 4};
  nd::array i = {3};
  EXPECT_ARRAY_EQ(3, nd::index(a, i));

  a = {{0, 1, 2}, {3, 4, 5}};
  i = {1, 1};
  EXPECT_ARRAY_EQ(4, nd::index(a, i));
}
