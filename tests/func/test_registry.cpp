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
  EXPECT_DOUBLE_EQ(sin(2.0f), af(2.0f).as<float>());
  // Implicit int -> double conversion
  EXPECT_DOUBLE_EQ(sin(3), af(3.0).as<double>());
  // Bool doesn't implicitly convert to float
  EXPECT_THROW(af(true), type_error);
  af = func::get_regfunction("cos");
  EXPECT_DOUBLE_EQ(cos(1.0), af(1.0).as<double>());
  af = func::get_regfunction("exp");
  EXPECT_DOUBLE_EQ(exp(1.0), af(1.0).as<double>());
}
