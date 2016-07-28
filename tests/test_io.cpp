//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/io.hpp>
#include <dynd_assertions.hpp>

using namespace std;
using namespace dynd;

TEST(Serialize, FixedDim) {
  EXPECT_ARRAY_EQ(bytes("\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00"),
                  nd::serialize(nd::array{0, 1, 2, 3, 4}));
}

TEST(Serialize, FixedDimFixedDim) {
  EXPECT_ARRAY_EQ(bytes("\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00"),
                  nd::serialize(nd::array{{0, 1}, {2, 3}}));
}
