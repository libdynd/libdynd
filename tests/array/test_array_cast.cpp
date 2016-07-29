//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dynd/array.hpp"
#include "dynd/exceptions.hpp"
#include <dynd/gtest.hpp>

using namespace std;
using namespace dynd;

TEST(ArrayCast, BasicCast) {
  nd::array n = 3.5;
  EXPECT_EQ(3.5f, nd::empty(ndt::make_type<float>()).assign(n));
}
