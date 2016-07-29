//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/gtest.hpp>
#include <dynd/search.hpp>

using namespace std;
using namespace dynd;

TEST(Search, BinarySearch) {
  EXPECT_ARRAY_VALS_EQ(1, nd::binary_search(nd::array{0, 1, 2}, 1));
  EXPECT_ARRAY_VALS_EQ(1, nd::binary_search(nd::array{5, 3, 1}, 3));
  EXPECT_ARRAY_VALS_EQ(-1, nd::binary_search(nd::array{5, 3, 1}, 10));
}
