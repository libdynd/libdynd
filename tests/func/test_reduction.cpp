//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/array.hpp>
#include <dynd/kernels/reduction_kernels.hpp>
#include <dynd/func/reduction.hpp>
#include <dynd/func/sum.hpp>
#include <dynd/json_parser.hpp>

#include "dynd_assertions.hpp"

using namespace std;
using namespace dynd;

TEST(Sum, 1D)
{
  // int32
  EXPECT_ARR_EQ(1, nd::sum(nd::array{1}));
  EXPECT_ARR_EQ(-1, nd::sum(nd::array{1, -2}));
  EXPECT_ARR_EQ(11, nd::sum(nd::array{1, -2, 12}));

  // int64
  EXPECT_ARR_EQ(1LL, nd::sum(nd::array{1LL}));
  EXPECT_ARR_EQ(-19999999999LL, nd::sum(nd::array{1LL, -20000000000LL}));
  EXPECT_ARR_EQ(-19999999987LL, nd::sum(nd::array{1LL, -20000000000LL, 12LL}));

  // float32
  EXPECT_ARR_EQ(1.25f, nd::sum(nd::array{1.25f}));
  EXPECT_ARR_EQ(-1.25f, nd::sum(nd::array{1.25f, -2.5f}));
  EXPECT_ARR_EQ(10.875f, nd::sum(nd::array{1.25f, -2.5f, 12.125f}));

  // float64
  EXPECT_ARR_EQ(1.25, nd::sum(nd::array{1.25}));
  EXPECT_ARR_EQ(-1.25, nd::sum(nd::array{1.25, -2.5}));
  EXPECT_ARR_EQ(10.875, nd::sum(nd::array{1.25, -2.5, 12.125}));

  // complex[float32]
  EXPECT_ARR_EQ(dynd::complex<float>(1.25f, -2.125f),
                nd::sum(nd::array{dynd::complex<float>(1.25f, -2.125f)}));
  EXPECT_ARR_EQ(dynd::complex<float>(-1.25f, -1.125f),
                nd::sum(nd::array{dynd::complex<float>(1.25f, -2.125f),
                                  dynd::complex<float>(-2.5f, 1.0f)}));
  EXPECT_ARR_EQ(dynd::complex<float>(10.875f, 12343.875f),
                nd::sum(nd::array{dynd::complex<float>(1.25f, -2.125f),
                                  dynd::complex<float>(-2.5f, 1.0f),
                                  dynd::complex<float>(12.125f, 12345.0f)}));

  // complex[float64]
  EXPECT_ARR_EQ(dynd::complex<double>(1.25, -2.125),
                nd::sum(nd::array{dynd::complex<double>(1.25, -2.125)}));
  EXPECT_ARR_EQ(dynd::complex<double>(-1.25, -1.125),
                nd::sum(nd::array{dynd::complex<double>(1.25, -2.125),
                                  dynd::complex<double>(-2.5, 1.0)}));
  EXPECT_ARR_EQ(dynd::complex<double>(10.875, 12343.875),
                nd::sum(nd::array{dynd::complex<double>(1.25, -2.125),
                                  dynd::complex<double>(-2.5, 1.0),
                                  dynd::complex<double>(12.125, 12345.0)}));
}

TEST(Reduction, BuiltinSum_Lift0D_NoIdentity)
{
  nd::callable f = nd::functional::reduction(
      nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARR_EQ(1.25, f(1.25));
}

TEST(Reduction, BuiltinSum_Lift0D_WithIdentity)
{
  nd::callable f = nd::functional::reduction(
      nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARR_EQ(100.0 + 1.25, f(1.25, kwds("identity", 100.0)));
}

TEST(Reduction, BuiltinSum_Lift1D_NoIdentity)
{
  nd::callable f = nd::functional::reduction(
      nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARR_EQ(1.5 - 22.0 + 3.75 + 1.125 - 3.375,
                f(initializer_list<double>{1.5, -22.0, 3.75, 1.125, -3.375}));
  EXPECT_ARR_EQ(3.75, f(initializer_list<double>{3.75}));
}

TEST(Reduction, BuiltinSum_Lift1D_WithIdentity)
{
  nd::callable f = nd::functional::reduction(
      nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARR_EQ(100.0 + 1.5 - 22.0 + 3.75 + 1.125 - 3.375,
                f(initializer_list<double>{1.5, -22., 3.75, 1.125, -3.375},
                  kwds("identity", 100.0)));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceReduce)
{
  nd::callable f = nd::functional::reduction(
      nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARR_EQ(1.5 + 2.0 + 7.0 - 2.25 + 7.0 + 2.125,
                f(initializer_list<initializer_list<double>>{
                    {1.5, 2.0, 7.0}, {-2.25, 7.0, 2.125}}));
  EXPECT_ARR_EQ(1.5 - 2.0,
                f(initializer_list<initializer_list<double>>{{1.5, -2.0}}));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceReduce_KeepDim)
{
  nd::callable f = nd::functional::reduction(
      nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARR_EQ(initializer_list<initializer_list<double>>{
                    {1.5 + 2.0 + 7.0 - 2.25 + 7.0 + 2.125}},
                f(initializer_list<initializer_list<double>>{
                      {1.5, 2.0, 7.0}, {-2.25, 7.0, 2.125}},
                  kwds("keepdims", true)));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_BroadcastReduce)
{
  nd::callable f = nd::functional::reduction(
      nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARR_EQ(
      initializer_list<double>({1.5 + 2.0 + 7.0, -2.25 + 7.0 + 2.125}),
      f(initializer_list<initializer_list<double>>{{1.5, 2.0, 7.0},
                                                   {-2.25, 7.0, 2.125}},
        kwds("axes", nd::array(initializer_list<int>{1}))));
  EXPECT_ARR_EQ(initializer_list<double>{1.5 - 2.0},
                f(initializer_list<initializer_list<double>>{{1.5, -2.0}},
                  kwds("axes", nd::array(initializer_list<int>{1}))));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_BroadcastReduce_KeepDim)
{
  nd::callable f = nd::functional::reduction(
      nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARR_EQ(
      (initializer_list<initializer_list<double>>{{1.5 + 2.0 + 7.0},
                                                  {-2.25 + 7.0 + 2.125}}),
      f(initializer_list<initializer_list<double>>{{1.5, 2.0, 7.0},
                                                   {-2.25, 7.0, 2.125}},
        kwds("axes", nd::array(initializer_list<int>{1}), "keepdims", true)));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceBroadcast)
{
  nd::callable f = nd::functional::reduction(
      nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARR_EQ((initializer_list<double>{1.5 - 2.25, 2.0 + 7.0, 7.0 + 2.125}),
                f(initializer_list<initializer_list<double>>{
                      {1.5, 2.0, 7.0}, {-2.25, 7.0, 2.125}},
                  kwds("axes", nd::array(initializer_list<int>{0}))));
  EXPECT_ARR_EQ((initializer_list<double>{1.5, -2.0}),
                f(initializer_list<initializer_list<double>>{{1.5, -2.0}},
                  kwds("axes", nd::array(initializer_list<int>{0}))));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceBroadcast_KeepDim)
{
  nd::callable f = nd::functional::reduction(
      nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARR_EQ(
      (initializer_list<initializer_list<double>>{
          {1.5 - 2.25, 2.0 + 7.0, 7.0 + 2.125}}),
      f(initializer_list<initializer_list<double>>{{1.5, 2.0, 7.0},
                                                   {-2.25, 7.0, 2.125}},
        kwds("axes", nd::array(initializer_list<int>{0}), "keepdims", true)));
}

TEST(Reduction, BuiltinSum_Lift3D_StridedStridedStrided_ReduceReduceReduce)
{
  nd::callable f = nd::functional::reduction(
      nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARR_EQ(1.5 - 2.375 + 2.0 + 1.25 + 7.0 - 0.5 - 2.25 + 1.0 + 7.0 +
                    2.125 + 0.25,
                f(initializer_list<initializer_list<initializer_list<double>>>{
                    {{1.5, -2.375}, {2, 1.25}, {7, -0.5}},
                    {{-2.25, 1}, {7, 0}, {2.125, 0.25}}}));
}

TEST(Reduction, BuiltinSum_Lift3D_StridedStridedStrided_BroadcastReduceReduce)
{
  nd::callable f = nd::functional::reduction(
      nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARR_EQ((initializer_list<double>{1.5 - 2.375 + 2.0 + 1.25 + 7.0 - 0.5,
                                          -2.25 + 1.0 + 7.0 + 2.125 + 0.25}),
                f(initializer_list<initializer_list<initializer_list<double>>>{
                      {{1.5, -2.375}, {2.0, 1.25}, {7.0, -0.5}},
                      {{-2.25, 1.0}, {7.0, 0.0}, {2.125, 0.25}}},
                  kwds("axes", nd::array({1, 2}))));
}

TEST(Reduction, BuiltinSum_Lift3D_StridedStridedStrided_ReduceBroadcastReduce)
{
  nd::callable f = nd::functional::reduction(
      nd::functional::apply([](double x, double y) { return x + y; }));

  EXPECT_ARR_EQ(
      (initializer_list<double>{1.5 - 2.375 - 2.25 + 1.0, 2.0 + 1.25 + 7.0,
                                7.0 - 0.5 + 2.125 + 0.25}),
      f(initializer_list<initializer_list<initializer_list<double>>>{
            {{1.5, -2.375}, {2.0, 1.25}, {7.0, -0.5}},
            {{-2.25, 1.0}, {7.0, 0.0}, {2.125, 0.25}}},
        kwds("axes", nd::array({0, 2}))));
}
