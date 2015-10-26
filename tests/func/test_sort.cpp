//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/sort.hpp>

#include "dynd_assertions.hpp"

using namespace std;
using namespace dynd;

TEST(Sort, 1D)
{
  //  nd::array a = {19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  nd::array a = {2, 1, 0};
  nd::sort(a);
  EXPECT_ARRAY_EQ((nd::array{0, 1, 2}), a);
}
