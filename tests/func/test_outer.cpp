//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../dynd_assertions.hpp"
#include "inc_gtest.hpp"

#include <dynd/functional.hpp>

using namespace std;
using namespace dynd;

TEST(Outer, 1D) {
  nd::callable f = nd::functional::outer([](int x) { return x; });
  EXPECT_ARRAY_EQ(nd::array({0, 1}), f(nd::array{0, 1}));
}

TEST(Outer, 2D) {
  nd::callable f = nd::functional::outer([](int x, int y) { return x + y; });
  EXPECT_ARRAY_EQ(nd::array({{1, 3}, {3, 5}}), f(nd::array{0, 2}, nd::array{1, 3}));
}

TEST(Outer, 3D) {
  nd::callable f = nd::functional::outer([](int x, int y, int z) { return x + y + z; });
  EXPECT_ARRAY_EQ(nd::array({{{3, 6}, {6, 9}}, {{6, 9}, {9, 12}}}),
                  f(nd::array{0, 3}, nd::array{1, 4}, nd::array{2, 5}));
}

/*
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
*/
