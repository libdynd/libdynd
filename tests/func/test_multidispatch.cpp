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

#include <dynd/functional.hpp>
#include <dynd/random.hpp>

using namespace std;
using namespace dynd;

/*
int func0(int, float, double) { return 0; }
int func1(int, double, double) { return 1; }
int func2(int, float, float) { return 2; }
int func3(float, int, int) { return 3; }
int func4(int16_t, float, double) { return 4; }
int func5(int16_t, float, float) { return 5; }

double manip0(int x, int y) { return x + y; };
double manip1(double x, double y) { return x - y; };

// TODO: Reenable tests involving float16

TEST(MultiDispatchCallable, Ambiguous)
{
  vector<nd::callable> funcs;
  funcs.push_back(nd::functional::apply(&func0));
  funcs.push_back(nd::functional::apply(&func1));
  funcs.push_back(nd::functional::apply(&func2));
  funcs.push_back(nd::functional::apply(&func3));
  funcs.push_back(nd::functional::apply(&func4));

  EXPECT_THROW(nd::functional::old_multidispatch(funcs.size(), &funcs[0]), invalid_argument);

  funcs.push_back(nd::functional::apply(&func5));
}

TEST(MultiDispatchCallable, ExactSignatures)
{
  vector<nd::callable> funcs;
  funcs.push_back(nd::functional::apply(&func0));
  funcs.push_back(nd::functional::apply(&func1));
  funcs.push_back(nd::functional::apply(&func2));
  funcs.push_back(nd::functional::apply(&func3));
  funcs.push_back(nd::functional::apply(&func4));
  funcs.push_back(nd::functional::apply(&func5));

  nd::callable af = nd::functional::old_multidispatch(funcs.size(), &funcs[0]);

  EXPECT_ARRAY_EQ(0, af(1, 1.f, 1.0).as<int>());
  EXPECT_ARRAY_EQ(1, af(1, 1.0, 1.0).as<int>());
  EXPECT_ARRAY_EQ(2, af(1, 1.f, 1.f).as<int>());
  EXPECT_ARRAY_EQ(3, af(1.f, 1, 1).as<int>());
  EXPECT_ARRAY_EQ(4, af((int16_t)1, 1.f, 1.0).as<int>());
  EXPECT_ARRAY_EQ(5, af((int16_t)1, 1.f, 1.f).as<int>());
}

TEST(MultiDispatchCallable, PromoteToSignature)
{
  vector<nd::callable> funcs;
  funcs.push_back(nd::functional::apply(&func0));
  funcs.push_back(nd::functional::apply(&func1));
  funcs.push_back(nd::functional::apply(&func2));
  funcs.push_back(nd::functional::apply(&func3));
  funcs.push_back(nd::functional::apply(&func4));
  funcs.push_back(nd::functional::apply(&func5));

  nd::callable af = nd::functional::old_multidispatch(funcs.size(), &funcs[0]);

  //  EXPECT_EQ(0, af(1, float16(1.f), 1.0).as<int>());
  EXPECT_EQ(1, af(1, 1.0, 1.f).as<int>());
  //  EXPECT_EQ(2, af(1, 1.f, float16(1.f)).as<int>());
  EXPECT_EQ(3, af(1.f, 1, (int16_t)1).as<int>());
  EXPECT_EQ(4, af((int8_t)1, 1.f, 1.0).as<int>());
  EXPECT_EQ(5, af((int8_t)1, 1.f, 1.f).as<int>());
}

TEST(MultiDispatchCallable, Values)
{
  vector<nd::callable> funcs;
  funcs.push_back(nd::functional::apply(&manip0));
  funcs.push_back(nd::functional::apply(&manip1));
  nd::callable af = nd::functional::elwise(nd::functional::old_multidispatch(funcs.size(), &funcs[0]));
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
*/

/*
TEST(Multidispatch, Unary)
{
  nd::callable f0([](int) { return 0; });
  nd::callable f1([](float) { return 1; });
  nd::callable f2([](double) { return 2; });

  nd::callable f = nd::functional::multidispatch({f0, f1, f2});
  EXPECT_EQ(f0, f.specialize(ndt::make_type<int>(), {ndt::make_type<int>()}));
  EXPECT_EQ(f1, f.specialize(ndt::make_type<int>(), {ndt::make_type<float>()}));
  EXPECT_EQ(f2, f.specialize(ndt::make_type<int>(), {ndt::make_type<double>()}));

    EXPECT_ARRAY_EQ(0, func(int32()));
    EXPECT_ARRAY_EQ(1, func(float32()));
    EXPECT_ARRAY_EQ(2, func(float64()));
    EXPECT_THROW(func(int64()), runtime_error);
    EXPECT_THROW(func(float16()), runtime_error);

  f = nd::functional::multidispatch(ndt::type("(Any) -> Any"), {f0, f1, f2});
  EXPECT_EQ(f0, f.specialize(ndt::make_type<int>(), {ndt::make_type<int>()}));
  EXPECT_EQ(f1, f.specialize(ndt::make_type<int>(), {ndt::make_type<float>()}));
  EXPECT_EQ(f2, f.specialize(ndt::make_type<int>(), {ndt::make_type<double>()}));

  EXPECT_ARRAY_EQ(0, f(int()));
  EXPECT_ARRAY_EQ(1, f(float()));
  EXPECT_ARRAY_EQ(2, f(double()));
  EXPECT_THROW(f(int64()), runtime_error);
  EXPECT_THROW(f(float16()), runtime_error);
}

TEST(Multidispatch, UnaryWithPermutation)
{
  nd::callable func0 = nd::functional::apply([](int32) { return 0; });
  nd::callable func1 = nd::functional::apply([](float32) { return 1; });
  nd::callable func2 = nd::functional::apply([](float64) { return 2; });

  nd::callable func = nd::functional::multidispatch({func0, func1, func2}, {0});
  EXPECT_ARRAY_EQ(0, func(int32()));
  EXPECT_ARRAY_EQ(1, func(float32()));
  EXPECT_ARRAY_EQ(2, func(float64()));
  EXPECT_THROW(func(int64()), runtime_error);
  EXPECT_THROW(func(float16()), runtime_error);
}

TEST(Multidispatch, Binary)
{
  nd::callable func0 = nd::functional::apply([](int32, int32) { return 0; });
  nd::callable func1 = nd::functional::apply([](int32, float32) { return 1; });
  nd::callable func2 = nd::functional::apply([](int32, float64) { return 2; });
  nd::callable func3 = nd::functional::apply([](float32, int32) { return 3; });
  nd::callable func4 = nd::functional::apply([](float64, float32) { return 4; });
  nd::callable func5 = nd::functional::apply([](float64, float64) { return 5; });

  nd::callable func = nd::functional::multidispatch({func0, func1, func2, func3, func4, func5});
    EXPECT_ARRAY_EQ(0, func(int32(), int32()));
    EXPECT_ARRAY_EQ(1, func(int32(), float32()));
    EXPECT_ARRAY_EQ(2, func(int32(), float64()));
    EXPECT_ARRAY_EQ(3, func(float32(), int32()));
    EXPECT_ARRAY_EQ(4, func(float64(), float32()));
    EXPECT_ARRAY_EQ(5, func(float64(), float64()));
    EXPECT_THROW(func(int32()), runtime_error);
    EXPECT_THROW(func(float32()), runtime_error);
    EXPECT_THROW(func(int64()), runtime_error);

  func = nd::functional::multidispatch(ndt::type("(Any, Any) -> Any"), {func0, func1, func2, func3, func4, func5});
  EXPECT_ARRAY_EQ(0, func(int32(), int32()));
  EXPECT_ARRAY_EQ(1, func(int32(), float32()));
  EXPECT_ARRAY_EQ(2, func(int32(), float64()));
  EXPECT_ARRAY_EQ(3, func(float32(), int32()));
  EXPECT_ARRAY_EQ(4, func(float64(), float32()));
  EXPECT_ARRAY_EQ(5, func(float64(), float64()));
}

TEST(Multidispatch, BinaryWithPermutation)
{
  nd::callable func0 = nd::functional::apply([](int32, int32) { return 0; });
  nd::callable func1 = nd::functional::apply([](int32, float32) { return 1; });
  nd::callable func2 = nd::functional::apply([](int32, float64) { return 2; });

  nd::callable func = nd::functional::multidispatch({func0, func1, func2}, {1});
  EXPECT_ARRAY_EQ(0, func(int32(), int32()));
  EXPECT_ARRAY_EQ(1, func(int32(), float32()));
  EXPECT_ARRAY_EQ(2, func(int32(), float64()));
  EXPECT_THROW(func(int32(), int64()), runtime_error);
  EXPECT_THROW(func(int32(), float16()), runtime_error);
}
*/
