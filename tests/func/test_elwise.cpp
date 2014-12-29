//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/func/apply_arrfunc.hpp>
#include <dynd/func/elwise.hpp>

struct anon_func {
  int operator()(int x, int y) { return x + y; }
};

TEST(Elwise, Untitled)
{
  nd::arrfunc af;

  nd::array a = parse_json("3 * int", "[0, 1, 2]");
  nd::array b = parse_json("3 * int", "[3, 4, 5]");

  af = nd::apply::make<anon_func>();
  EXPECT_ARR_EQ(nd::array({3, 5, 7}), nd::elwise(a, b, kwds("func", af)));
}