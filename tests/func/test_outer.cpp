//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/func/apply.hpp>
#include <dynd/func/outer.hpp>

namespace {

int func0(int x, int y, int z) {
  return x + y + z;
}

} // unnamed namespace

TEST(Outer, Simple)
{
  nd::arrfunc af = nd::functional::outer(nd::functional::apply(&func0));
  std::cout << af(nd::array({0, 1}), nd::array({3, 4}), nd::array({5, 6})) << std::endl;

  std::exit(-1);
}