//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "../dynd_assertions.hpp"

#include <dynd/functional.hpp>

using namespace std;
using namespace dynd;

TEST(Functional, LeftCompound)
{
  nd::callable f = nd::functional::left_compound(nd::functional::apply([](int x, int y) { return x + y; }));

  nd::array y = nd::empty(ndt::make_type<int>());
  y.assign(3);
  EXPECT_ARRAY_EQ(8, f({5}, {{"dst", y}}));
}

TEST(Functional, RightCompound)
{
  nd::callable f = nd::functional::right_compound(nd::functional::apply([](int x, int y) { return x - y; }));

  nd::array y = nd::empty(ndt::make_type<int>());
  y.assign(3);
  EXPECT_ARRAY_EQ(2, f({5}, {{"dst", y}}));
}
