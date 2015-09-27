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

#include <dynd/func/view.hpp>

using namespace std;
using namespace dynd;

TEST(View, Simple)
{
  nd::array a = 3;
  nd::view(a);
//  EXPECT_ARRAY_EQ(3, nd::view(a));
}
