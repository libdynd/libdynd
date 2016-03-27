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

#include "dynd_assertions.hpp"

using namespace std;
using namespace dynd;

/*
TEST(Reduction, BuiltinSum_Lift0D_NoIdentity)
{
  nd::callable f = nd::functional::reduction(nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARRAY_EQ(1.25, f(1.25));
}
*/

/*
TEST(Reduction, BuiltinSum_Lift0D_WithIdentity)
{
  nd::callable f = nd::functional::reduction(
      nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARRAY_EQ(100.0 + 1.25, f({1.25}, {{"identity", 100.0}}));
}
*/

TEST(Reduction, BuiltinSum_Lift1D_NoIdentity)
{
  nd::callable f = nd::functional::reduction(nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARRAY_EQ(1.5 - 22.0 + 3.75 + 1.125 - 3.375, f(initializer_list<double>{1.5, -22.0, 3.75, 1.125, -3.375}));
  EXPECT_ARRAY_EQ(3.75, f(initializer_list<double>{3.75}));
}

TEST(Reduction, BuiltinSum_Lift1D_WithIdentity)
{
  nd::callable f = nd::functional::reduction(nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARRAY_EQ(100.0 + 1.5 - 22.0 + 3.75 + 1.125 - 3.375,
                  f({initializer_list<double>{1.5, -22., 3.75, 1.125, -3.375}}, {{"identity", 100.0}}));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceReduce)
{
  nd::callable f = nd::functional::reduction(nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARRAY_EQ(1.5 + 2.0 + 7.0 - 2.25 + 7.0 + 2.125,
                  f(initializer_list<initializer_list<double>>{{1.5, 2.0, 7.0}, {-2.25, 7.0, 2.125}}));
  EXPECT_ARRAY_EQ(1.5 - 2.0, f(initializer_list<initializer_list<double>>{{1.5, -2.0}}));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceReduce_KeepDim)
{
  nd::callable f = nd::functional::reduction(nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARRAY_EQ(
      initializer_list<initializer_list<double>>{{1.5 + 2.0 + 7.0 - 2.25 + 7.0 + 2.125}},
      f({initializer_list<initializer_list<double>>{{1.5, 2.0, 7.0}, {-2.25, 7.0, 2.125}}}, {{"keepdims", true}}));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_BroadcastReduce)
{
  nd::callable f = nd::functional::reduction(nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARRAY_EQ(initializer_list<double>({1.5 + 2.0 + 7.0, -2.25 + 7.0 + 2.125}),
                  f({initializer_list<initializer_list<double>>{{1.5, 2.0, 7.0}, {-2.25, 7.0, 2.125}}},
                    {{"axes", nd::array(initializer_list<int>{1})}}));
  EXPECT_ARRAY_EQ(initializer_list<double>{1.5 - 2.0}, f({initializer_list<initializer_list<double>>{{1.5, -2.0}}},
                                                         {{"axes", nd::array(initializer_list<int>{1})}}));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_BroadcastReduce_KeepDim)
{
  nd::callable f = nd::functional::reduction(nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARRAY_EQ((initializer_list<initializer_list<double>>{{1.5 + 2.0 + 7.0}, {-2.25 + 7.0 + 2.125}}),
                  f({initializer_list<initializer_list<double>>{{1.5, 2.0, 7.0}, {-2.25, 7.0, 2.125}}},
                    {{"axes", nd::array(initializer_list<int>{1})}, {"keepdims", true}}));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceBroadcast)
{
  nd::callable f = nd::functional::reduction(nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARRAY_EQ((initializer_list<double>{1.5 - 2.25, 2.0 + 7.0, 7.0 + 2.125}),
                  f({initializer_list<initializer_list<double>>{{1.5, 2.0, 7.0}, {-2.25, 7.0, 2.125}}},
                    {{"axes", nd::array(initializer_list<int>{0})}}));
  EXPECT_ARRAY_EQ((initializer_list<double>{1.5, -2.0}), f({initializer_list<initializer_list<double>>{{1.5, -2.0}}},
                                                           {{"axes", nd::array(initializer_list<int>{0})}}));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceBroadcast_KeepDim)
{
  nd::callable f = nd::functional::reduction(nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARRAY_EQ((initializer_list<initializer_list<double>>{{1.5 - 2.25, 2.0 + 7.0, 7.0 + 2.125}}),
                  f({initializer_list<initializer_list<double>>{{1.5, 2.0, 7.0}, {-2.25, 7.0, 2.125}}},
                    {{"axes", nd::array(initializer_list<int>{0})}, {"keepdims", true}}));
}

TEST(Reduction, FixedVar)
{
  nd::callable f = nd::functional::reduction(nd::functional::apply([](double x, double y) { return x + y; }));

  nd::array a = nd::empty("3 * var * float64");
  a(0).vals() = {1, 2};
  a(1).vals() = {3, 4, 5};
  a(2).vals() = {6, 7, 8, 9};

  EXPECT_ARRAY_EQ(45.0, f(a));
}

TEST(Reduction, FixedVarWithAxes)
{
  nd::callable f = nd::functional::reduction(nd::functional::apply([](double x, double y) { return x + y; }));

  nd::array a = nd::empty("3 * var * float64");
  a(0).vals() = {1, 2};
  a(1).vals() = {3, 4, 5};
  a(2).vals() = {6, 7, 8, 9};

  EXPECT_ARRAY_EQ((nd::array{3.0, 12.0, 30.0}), f({a}, {{"axes", nd::array{1}}}));
}

TEST(Reduction, BuiltinSum_Lift3D_StridedStridedStrided_ReduceReduceReduce)
{
  nd::callable f = nd::functional::reduction(nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARRAY_EQ(1.5 - 2.375 + 2.0 + 1.25 + 7.0 - 0.5 - 2.25 + 1.0 + 7.0 + 2.125 + 0.25,
                  f(initializer_list<initializer_list<initializer_list<double>>>{{{1.5, -2.375}, {2, 1.25}, {7, -0.5}},
                                                                                 {{-2.25, 1}, {7, 0}, {2.125, 0.25}}}));
}

TEST(Reduction, BuiltinSum_Lift3D_StridedStridedStrided_BroadcastReduceReduce)
{
  nd::callable f = nd::functional::reduction(nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARRAY_EQ((nd::array{1.5 - 2.375 + 2.0 + 1.25 + 7.0 - 0.5, -2.25 + 1.0 + 7.0 + 2.125 + 0.25}),
                  f({nd::array{{{1.5, -2.375}, {2.0, 1.25}, {7.0, -0.5}}, {{-2.25, 1.0}, {7.0, 0.0}, {2.125, 0.25}}}},
                    {{"axes", {1, 2}}}));
}

TEST(Reduction, BuiltinSum_Lift3D_StridedStridedStrided_ReduceBroadcastReduce)
{
  nd::callable f = nd::functional::reduction(nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARRAY_EQ(
      (nd::array{1.5 - 2.375 - 2.25 + 1.0, 2.0 + 1.25 + 7.0, 7.0 - 0.5 + 2.125 + 0.25}),
      f({{{{1.5, -2.375}, {2.0, 1.25}, {7.0, -0.5}}, {{-2.25, 1.0}, {7.0, 0.0}, {2.125, 0.25}}}}, {{"axes", {0, 2}}}));
}

TEST(Reduction, Except)
{
  // Cannot have a null child
  EXPECT_THROW(nd::functional::reduction(nd::callable()), invalid_argument);

  // Cannot have a child with no arguments
  EXPECT_THROW(nd::functional::reduction(nd::functional::apply([]() { return 0; })), invalid_argument);

  // Cannot have a child with more than two arguments
  EXPECT_THROW(nd::functional::reduction(nd::functional::apply([](int x, int y, int z) { return x ? y : z; })),
               invalid_argument);
  EXPECT_THROW(nd::functional::reduction(nd::functional::apply([](int x, double y, double z) { return x ? y : z; })),
               invalid_argument);
}
