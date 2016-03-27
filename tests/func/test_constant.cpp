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

#include <dynd/functional.hpp>

using namespace std;
using namespace dynd;

TEST(Functional, Constant)
{
  nd::callable f = nd::functional::constant(2);

  EXPECT_ARRAY_EQ(2, f(1));
  EXPECT_ARRAY_EQ(2, f(2.0));
  EXPECT_ARRAY_EQ(2, f(nd::array{0.0, 1.0, 2.0}));

  f = nd::functional::constant(10.0);

  EXPECT_ARRAY_EQ(10.0, f(1));
  EXPECT_ARRAY_EQ(10.0, f(2.0));
  EXPECT_ARRAY_EQ(10.0, f(nd::array{0.0, 1.0, 2.0}));
}
