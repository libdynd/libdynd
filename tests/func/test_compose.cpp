//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/assignment.hpp>
#include <dynd/callable.hpp>
#include <dynd/convert.hpp>
#include <dynd/functional.hpp>
#include <dynd/functional.hpp>
#include <dynd/index.hpp>
#include <dynd/types/fixed_string_type.hpp>

using namespace std;
using namespace dynd;

TEST(Compose, Simple) {
  reg_entry &reg = nd::registry();

  nd::callable composed = nd::functional::compose(nd::copy, reg["sin"].value(), ndt::make_type<double>());
  nd::array a = nd::empty(ndt::make_type<double>());
  composed({"0.0"}, {{"dst", a}});
  EXPECT_EQ(0., a.as<double>());
  composed({"1.5"}, {{"dst", a}});
  EXPECT_DOUBLE_EQ(sin(1.5), a.as<double>());
  composed({3.1}, {{"dst", a}});
  EXPECT_DOUBLE_EQ(sin(3.1), a.as<double>());
}

/*
TEST(Convert, Unary)
{
  nd::callable f = nd::functional::apply([](double x) { return x; });
  nd::callable g = nd::functional::convert(ndt::type("(float32) -> float64"), f);
}
*/
