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

#include <dynd/io.hpp>

using namespace std;
using namespace dynd;

TEST(Serialize, FixedDim)
{
  EXPECT_ARRAY_EQ(bytes("\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00"),
                  nd::serialize({{0, 1, 2, 3, 4}}, {{"identity", nd::empty(ndt::type("bytes"))}}));
}

TEST(Serialize, FixedDimFixedDim)
{
  EXPECT_ARRAY_EQ(bytes("\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00"),
                  nd::serialize({{{0, 1}, {2, 3}}}, {{"identity", nd::empty(ndt::type("bytes"))}}));
}
