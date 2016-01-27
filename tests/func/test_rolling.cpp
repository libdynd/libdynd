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

#include <dynd/func/rolling.hpp>
#include <dynd/func/mean.hpp>

using namespace std;
using namespace dynd;

/*
ToDo: Reenable this.

TEST(Rolling, BuiltinSum_Kernel)
{
  nd::callable sum_1d = kernels::make_builtin_sum1d_callable(float64_id);
  nd::callable rolling_sum = nd::functional::rolling(sum_1d, 4);

  double adata[] = {1, 3, 7, 2, 9, 4, -5, 100, 2, -20, 3, 9, 18};
  nd::array a = adata;
  nd::array b = rolling_sum(a);
  EXPECT_EQ(ndt::type("13 * real"), b.get_type());
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(dynd::isnan(b(i).as<double>()));
  }
  for (int i = 3, i_end = (int)b.get_dim_size(); i < i_end; ++i) {
    double s = 0;
    for (int j = i - 3; j <= i; ++j) {
      s += adata[j];
    }
    EXPECT_EQ(s, b(i).as<double>());
  }
}

TEST(Rolling, BuiltinMean_Kernel)
{
  nd::callable mean_1d = nd::make_builtin_mean1d_callable(float64_id, 0);
  nd::callable rolling_sum = nd::functional::rolling(mean_1d, 4);

  double adata[] = {1, 3, 7, 2, 9, 4, -5, 100, 2, -20, 3, 9, 18};
  nd::array a = adata;
  nd::array b = rolling_sum(a);
  EXPECT_EQ(ndt::type("13 * real"), b.get_type());
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(dynd::isnan(b(i).as<double>()));
  }
  for (int i = 3, i_end = (int)b.get_dim_size(); i < i_end; ++i) {
    double s = 0;
    for (int j = i - 3; j <= i; ++j) {
      s += adata[j];
    }
    EXPECT_EQ(s / 4, b(i).as<double>());
  }
}

*/
