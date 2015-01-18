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
#include <dynd/func/random.hpp>

static int func0(int x, int y, int z) { return x + y + z; }

TEST(Outer, Simple)
{
  nd::array x = nd::random::uniform(
      kwds("dst_tp", ndt::make_fixed_dim(10, ndt::make_type<int>())));
  nd::array y = nd::random::uniform(
      kwds("dst_tp", ndt::make_fixed_dim(10, ndt::make_type<int>())));
  nd::array z = nd::random::uniform(
      kwds("dst_tp", ndt::make_fixed_dim(10, ndt::make_type<int>())));

  nd::arrfunc af = nd::functional::outer(nd::functional::apply(&func0));
  nd::array res = af(x, y, z);

  for (intptr_t i = 0; i < x.get_dim_size(); ++i) {
    for (intptr_t j = 0; j < y.get_dim_size(); ++j) {
      for (intptr_t k = 0; k < z.get_dim_size(); ++k) {
        EXPECT_EQ(x(i).as<int>() + y(j).as<int>() + z(k).as<int>(),
                  res(i, j, k).as<int>());
      }
    }
  }
}