//
// Copyright (C) 2011-14 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/func/multidispatch_arrfunc.hpp>
#include <dynd/func/apply_arrfunc.hpp>
#include <dynd/func/lift_arrfunc.hpp>

using namespace std;
using namespace dynd;

int func0(int, float, double) { return 0; }
int func1(int, double, double) { return 1; }
int func2(int, float, float) { return 2; }
int func3(float, int, int) { return 3; }
int func4(int16_t, float, double) { return 4; }
int func5(int16_t, float, float) { return 5; }

double manip0(int x, int y) {return x + y;};
double manip1(double x, double y) { return x - y;};

TEST(MultiDispatchArrfunc, Ambiguous) {
  vector<nd::arrfunc> funcs;
  funcs.push_back(nd::make_apply_arrfunc(&func0));
  funcs.push_back(nd::make_apply_arrfunc(&func1));
  funcs.push_back(nd::make_apply_arrfunc(&func2));
  funcs.push_back(nd::make_apply_arrfunc(&func3));
  funcs.push_back(nd::make_apply_arrfunc(&func4));

  EXPECT_THROW(make_multidispatch_arrfunc(funcs.size(), &funcs[0]),
               invalid_argument);

  funcs.push_back(nd::make_apply_arrfunc(&func5));
}

TEST(MultiDispatchArrfunc, ExactSignatures) {
  vector<nd::arrfunc> funcs;
  funcs.push_back(nd::make_apply_arrfunc(&func0));
  funcs.push_back(nd::make_apply_arrfunc(&func1));
  funcs.push_back(nd::make_apply_arrfunc(&func2));
  funcs.push_back(nd::make_apply_arrfunc(&func3));
  funcs.push_back(nd::make_apply_arrfunc(&func4));
  funcs.push_back(nd::make_apply_arrfunc(&func5));

  nd::arrfunc af = make_multidispatch_arrfunc(funcs.size(), &funcs[0]);

  EXPECT_EQ(0, af(1, 1.f, 1.0).as<int>());
  EXPECT_EQ(1, af(1, 1.0, 1.0).as<int>());
  EXPECT_EQ(2, af(1, 1.f, 1.f).as<int>());
  EXPECT_EQ(3, af(1.f, 1, 1).as<int>());
  EXPECT_EQ(4, af((int16_t)1, 1.f, 1.0).as<int>());
  EXPECT_EQ(5, af((int16_t)1, 1.f, 1.f).as<int>());
}

TEST(MultiDispatchArrfunc, PromoteToSignature) {
  vector<nd::arrfunc> funcs;
  funcs.push_back(nd::make_apply_arrfunc(&func0));
  funcs.push_back(nd::make_apply_arrfunc(&func1));
  funcs.push_back(nd::make_apply_arrfunc(&func2));
  funcs.push_back(nd::make_apply_arrfunc(&func3));
  funcs.push_back(nd::make_apply_arrfunc(&func4));
  funcs.push_back(nd::make_apply_arrfunc(&func5));

  nd::arrfunc af = make_multidispatch_arrfunc(funcs.size(), &funcs[0]);

  EXPECT_EQ(0, af(1, dynd_float16(1.f), 1.0).as<int>());
  EXPECT_EQ(1, af(1, 1.0, 1.f).as<int>());
  EXPECT_EQ(2, af(1, 1.f, dynd_float16(1.f)).as<int>());
  EXPECT_EQ(3, af(1.f, 1, (int16_t)1).as<int>());
  EXPECT_EQ(4, af((int8_t)1, 1.f, 1.0).as<int>());
  EXPECT_EQ(5, af((int8_t)1, 1.f, 1.f).as<int>());
}

TEST(MultiDispatchArrfunc, Values) {
  vector<nd::arrfunc> funcs;
  funcs.push_back(nd::make_apply_arrfunc(&manip0));
  funcs.push_back(nd::make_apply_arrfunc(&manip1));
  nd::arrfunc af =
      lift_arrfunc(make_multidispatch_arrfunc(funcs.size(), &funcs[0]));
  nd::array a, b, c;

  // Exactly match (int, int) -> real
  a = parse_json("3 * int", "[1, 3, 5]");
  b = parse_json("3 * int", "[2, 5, 1]");
  c = af(a, b);
  EXPECT_EQ(ndt::type("3 * float64"), c.get_type());
  EXPECT_JSON_EQ_ARR("[3, 8, 6]", c);

  // Exactly match (real, real) -> real
  a = parse_json("3 * real", "[1, 3, 5]");
  b = parse_json("3 * real", "[2, 5, 1]");
  c = af(a, b);
  EXPECT_EQ(ndt::type("3 * float64"), c.get_type());
  EXPECT_JSON_EQ_ARR("[-1, -2, 4]", c);

  // Promote to (int, int) -> real
  a = parse_json("3 * int16", "[1, 3, 5]");
  b = parse_json("3 * int8", "[2, 5, 1]");
  c = af(a, b);
  EXPECT_EQ(ndt::type("3 * float64"), c.get_type());
  EXPECT_JSON_EQ_ARR("[3, 8, 6]", c);

  // Promote to (real, real) -> real
  a = parse_json("3 * int16", "[1, 3, 5]");
  b = parse_json("3 * float16", "[2, 5, 1]");
  c = af(a, b);
  EXPECT_EQ(ndt::type("3 * float64"), c.get_type());
  EXPECT_JSON_EQ_ARR("[-1, -2, 4]", c);
}

TEST(MultiDispatchArrfunc, Dims)
{
  vector<nd::arrfunc> funcs;
  // Instead of making a multidispatch arrfunc, then lifting it,
  // we lift multiple arrfuncs, then make a multidispatch arrfunc from them.
  funcs.push_back(lift_arrfunc(nd::make_apply_arrfunc(&manip0)));
  funcs.push_back(lift_arrfunc(nd::make_apply_arrfunc(&manip1)));
  nd::arrfunc af = make_multidispatch_arrfunc(funcs.size(), &funcs[0]);
  nd::array a, b, c;

  // Exactly match (int, int) -> real
  a = parse_json("3 * int", "[1, 3, 5]");
  b = parse_json("3 * int", "[2, 5, 1]");
  c = af(a, b);
  EXPECT_EQ(ndt::type("3 * float64"), c.get_type());
  EXPECT_JSON_EQ_ARR("[3, 8, 6]", c);
}
