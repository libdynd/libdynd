//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/func/arrfunc_registry.hpp>

using namespace std;
using namespace dynd;

TEST(ArrFuncRegistry, Unary) {
  nd::arrfunc af;
  af = func::get_regfunction("sin");
  EXPECT_DOUBLE_EQ(sin(1.0), af(1.0).as<double>());
}
