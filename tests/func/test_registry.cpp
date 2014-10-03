//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <math.h>

#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/func/arrfunc_registry.hpp>

using namespace std;
using namespace dynd;

TEST(ArrFuncRegistry, Dispatch) {
  nd::arrfunc af;
  af = func::get_regfunction("sin");
  // These are exact overloads of ``sin``
  EXPECT_DOUBLE_EQ(sin(1.0), af(1.0).as<double>());
  EXPECT_FLOAT_EQ(sin(2.0f), af(2.0f).as<float>());
  // Implicit int -> double conversion
  EXPECT_DOUBLE_EQ(sin(3.0), af(3).as<double>());
  EXPECT_DOUBLE_EQ(sin(4.0), af(4u).as<double>());
  // Bool doesn't implicitly convert to float
  EXPECT_THROW(af(true), type_error);
}

TEST(ArrFuncRegistry, Arithmetic) {
  // Simple sanity checks
  nd::arrfunc af;
  af = func::get_regfunction("add");
  EXPECT_EQ(ndt::type("int32"), af((int8_t)3, (int8_t)4).get_type());
  EXPECT_EQ(8, af(3, 5).as<int>());
  EXPECT_EQ(ndt::type("float32"), af(3.5f, 5.25f).get_type());
  EXPECT_EQ(8.75, af(3.5f, 5.25f).as<float>());
  af = func::get_regfunction("subtract");
  EXPECT_EQ(ndt::type("float64"), af(3.5, 4).get_type());
  EXPECT_EQ(-0.5, af(3.5, 4).as<double>());
  af = func::get_regfunction("multiply");
  EXPECT_EQ(ndt::type("float32"), af(3.5f, (int8_t)4).get_type());
  EXPECT_EQ(14, af(3.5f, (int8_t)4).as<float>());
  af = func::get_regfunction("divide");
  EXPECT_EQ(ndt::type("float64"), af(12.0, (int8_t)4).get_type());
  EXPECT_EQ(3, af(12.0, (int8_t)4).as<double>());
}

TEST(ArrFuncRegistry, Trig) {
  // Simple sanity checks
  nd::arrfunc af;
  af = func::get_regfunction("sin");
  EXPECT_FLOAT_EQ(sinf(2.0f), af(2.0f).as<float>());
  EXPECT_DOUBLE_EQ(sin(1.0), af(1.0).as<double>());
  af = func::get_regfunction("cos");
  EXPECT_FLOAT_EQ(cosf(1.f), af(1.f).as<float>());
  EXPECT_DOUBLE_EQ(cos(1.0), af(1.0).as<double>());
  af = func::get_regfunction("tan");
  EXPECT_FLOAT_EQ(tanf(1.f), af(1.f).as<float>());
  EXPECT_DOUBLE_EQ(tan(1.0), af(1.0).as<double>());
  af = func::get_regfunction("exp");
  EXPECT_FLOAT_EQ(expf(1.f), af(1.f).as<float>());
  EXPECT_DOUBLE_EQ(exp(1.0), af(1.0).as<double>());
  af = func::get_regfunction("arcsin");
  EXPECT_FLOAT_EQ(asinf(0.4f), af(0.4f).as<float>());
  EXPECT_DOUBLE_EQ(asin(1.0), af(1.0).as<double>());
  af = func::get_regfunction("arccos");
  EXPECT_FLOAT_EQ(acosf(1.f), af(1.f).as<float>());
  EXPECT_DOUBLE_EQ(acos(1.0), af(1.0).as<double>());
  af = func::get_regfunction("arctan");
  EXPECT_FLOAT_EQ(atanf(1.f), af(1.f).as<float>());
  EXPECT_DOUBLE_EQ(atan(1.0), af(1.0).as<double>());
  af = func::get_regfunction("arctan2");
  EXPECT_FLOAT_EQ(atan2f(1.f, 2.f), af(1.f, 2.f).as<float>());
  EXPECT_DOUBLE_EQ(atan2(1.0, 2.0), af(1.0, 2.0).as<double>());
  af = func::get_regfunction("hypot");
  EXPECT_FLOAT_EQ(5, af(3.f, 4.f).as<float>());
  EXPECT_DOUBLE_EQ(hypot(3.0, 4.5), af(3.0, 4.5).as<double>());
  af = func::get_regfunction("sinh");
  EXPECT_FLOAT_EQ(sinhf(2.0f), af(2.0f).as<float>());
  EXPECT_DOUBLE_EQ(sinh(1.0), af(1.0).as<double>());
  af = func::get_regfunction("cosh");
  EXPECT_FLOAT_EQ(coshf(1.f), af(1.f).as<float>());
  EXPECT_DOUBLE_EQ(cosh(1.0), af(1.0).as<double>());
  af = func::get_regfunction("tanh");
  EXPECT_FLOAT_EQ(tanhf(1.f), af(1.f).as<float>());
  EXPECT_DOUBLE_EQ(tanh(1.0), af(1.0).as<double>());
  af = func::get_regfunction("power");
  EXPECT_FLOAT_EQ(powf(1.5f, 2.25f), af(1.5f, 2.25f).as<float>());
  EXPECT_DOUBLE_EQ(pow(1.5, 2.25), af(1.5, 2.25).as<double>());
}
