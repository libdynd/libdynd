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

#include <dynd/range.hpp>

using namespace std;
using namespace dynd;

TEST(Range, Range) {
  EXPECT_ARRAY_EQ(nd::array({0, 1, 2, 3, 4}), nd::range({}, {{"stop", 5}}));
  EXPECT_ARRAY_EQ(nd::array({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), nd::range({}, {{"stop", 10}}));
}
