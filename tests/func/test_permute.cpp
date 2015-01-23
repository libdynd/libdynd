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
#include <dynd/func/permute.hpp>

static void func0(double &x, int y) { x = 3.5 * y; }

TEST(Permute, ReturnType)
{
  nd::arrfunc af = nd::functional::apply(&func0);
  nd::arrfunc paf = nd::functional::permute(af, {-1, 0});

  nd::array res = nd::empty(ndt::make_type<double>());
  af(res, 15);
  EXPECT_EQ(res, paf(15));
}