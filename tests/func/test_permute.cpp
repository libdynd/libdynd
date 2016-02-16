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

#include <dynd/func/elwise.hpp>
#include <dynd/func/outer.hpp>
#include <dynd/func/permute.hpp>

using namespace std;
using namespace dynd;

/*
static void func0(double &x, int y) { x = 3.5 * y; }

static void func1(double (&res)[3], double x, double y, double z)
{
  res[0] = y - z;
  res[1] = x - z;
  res[2] = y - x;
}
*/

/*
TEST(Permute, ReturnType)
{
  nd::callable af = nd::functional::apply(&func0);
  nd::callable paf = nd::functional::permute(af, {-1, 0});

  nd::array res = nd::empty(ndt::make_type<double>());
  af(res, 15);
  paf(15);
  EXPECT_TRUE((res == paf(15)).as<bool>());

  af = nd::functional::apply(&func1);
  paf = nd::functional::permute(af, {-1, 0, 1, 2});
  res = nd::empty(paf.get_type()->get_return_type());
  af(res, 5.0, 10.0, 1.0);
  EXPECT_ARRAY_EQ(res, paf(5.0, 10.0, 1.0));
}
*/
