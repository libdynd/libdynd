//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/functional.hpp>
#include <dynd/func/random.hpp>

using namespace std;
using namespace dynd;

static double func0(double x, double y, double z) { return x + y + z; }

TEST(Outer, 1D)
{
  nd::array x, y, z, res;
  nd::callable af = nd::functional::outer(nd::functional::apply(&func0));

  x = nd::random::uniform({}, {{"dst_tp", ndt::make_fixed_dim(10, ndt::make_type<double>())}});
  y = nd::random::uniform({}, {{"dst_tp", ndt::make_fixed_dim(10, ndt::make_type<double>())}});
  z = nd::random::uniform({}, {{"dst_tp", ndt::make_fixed_dim(10, ndt::make_type<double>())}});

  res = af(x, y, z);
  for (intptr_t i = 0; i < x.get_dim_size(); ++i) {
    for (intptr_t j = 0; j < y.get_dim_size(); ++j) {
      for (intptr_t k = 0; k < z.get_dim_size(); ++k) {
        EXPECT_EQ(x(i).as<double>() + y(j).as<double>() + z(k).as<double>(), res(i, j, k).as<double>());
      }
    }
  }

  x = nd::random::uniform({}, {{"dst_tp", ndt::make_fixed_dim(4, ndt::make_type<double>())}});
  y = nd::random::uniform({}, {{"dst_tp", ndt::make_fixed_dim(16, ndt::make_type<double>())}});
  z = nd::random::uniform({}, {{"dst_tp", ndt::make_fixed_dim(8, ndt::make_type<double>())}});

  res = af(x, y, z);
  for (intptr_t i = 0; i < x.get_dim_size(); ++i) {
    for (intptr_t j = 0; j < y.get_dim_size(); ++j) {
      for (intptr_t k = 0; k < z.get_dim_size(); ++k) {
        EXPECT_EQ(x(i).as<double>() + y(j).as<double>() + z(k).as<double>(), res(i, j, k).as<double>());
      }
    }
  }
}
