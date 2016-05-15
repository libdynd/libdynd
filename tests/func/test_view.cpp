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

#include <dynd/view.hpp>

using namespace std;
using namespace dynd;

/*
TEST(View, Simple)
{
  nd::array a = 3;
  EXPECT_ARRAY_EQ(3, nd::view(a));
}

TEST(View_, FixedDim)
{
  nd::array a{0, 1, 2, 3, 4};
  EXPECT_ARRAY_EQ(a, nd::view(a));
}
*/
