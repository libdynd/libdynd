//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dynd_assertions.hpp"
#include "inc_gtest.hpp"

#include <dynd/limits.hpp>

using namespace std;
using namespace dynd;

TEST(Limits, Min) {
  EXPECT_ARRAY_EQ(numeric_limits<int>::lowest(), nd::limits::min({}, {{"dst_tp", ndt::make_type<int>()}}));

  EXPECT_ARRAY_EQ(numeric_limits<float>::lowest(), nd::limits::min({}, {{"dst_tp", ndt::make_type<float>()}}));
  EXPECT_ARRAY_EQ(numeric_limits<double>::lowest(), nd::limits::min({}, {{"dst_tp", ndt::make_type<double>()}}));
}
