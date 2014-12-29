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
  nd::array a = parse_json("3 * int", "[0, 1, 2]");
  nd::array b = parse_json("3 * int", "[3, 4, 5]");

  nd::arrfunc af = nd::apply::make<kernel_request_host, anon_func>();
  std::cout << nd::elwise.get_funcproto() << std::endl;

  nd::array c = nd::elwise(a, b, kwds("func", af));
  std::cout << c << std::endl;

/*
  nd::arrfunc laf = lift_arrfunc(af);
  nd::array a = parse_json("3 * int", "[0, 1, 2]");
  nd::array b = parse_json("3 * int", "[3, 4, 5]");
  nd::array c = parse_json("3 * int", "[3, 5, 7]");
  EXPECT_ARR_EQ(c, laf(a, b));
*/

//  std::exit(-1);
}