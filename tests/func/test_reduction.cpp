//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

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

TEST(Reduction, BuiltinSum_Lift1D_NoIdentity) {
  nd::callable f =
      nd::functional::reduction([] { return 0.0; }, [](const return_wrapper<double> &res, double x) { res += x; });
  EXPECT_ARRAY_EQ(1.5 - 22.0 + 3.75 + 1.125 - 3.375, f(nd::array{1.5, -22.0, 3.75, 1.125, -3.375}));
  EXPECT_ARRAY_EQ(3.75, f(nd::array{3.75}));

  f = nd::functional::reduction([] { return 0; },
                                [](const return_wrapper<int> &res, int x, int y) { res += max(x, y); });
  EXPECT_ARRAY_EQ(39, f(nd::array{4, -1, 7, 9, 2}, nd::array{3, 8, 0, 5, 11}));
}

TEST(Reduction, BuiltinSum_Lift1D_WithIdentity) {
  nd::callable f =
      nd::functional::reduction([] { return 100.0; }, [](const return_wrapper<double> &res, double x) { res += x; });
  EXPECT_ARRAY_EQ(100.0 + 1.5 - 22.0 + 3.75 + 1.125 - 3.375, f(nd::array{1.5, -22., 3.75, 1.125, -3.375}));

  f = nd::functional::reduction([] { return 100; },
                                [](const return_wrapper<int> &res, int x, int y) { res += max(x, y); });
  EXPECT_ARRAY_EQ(139, f(nd::array{4, -1, 7, 9, 2}, nd::array{3, 8, 0, 5, 11}));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceReduce) {
  nd::callable f =
      nd::functional::reduction([] { return 0.0; }, [](const return_wrapper<double> &res, double x) { res += x; });
  EXPECT_ARRAY_EQ(1.5 + 2.0 + 7.0 - 2.25 + 7.0 + 2.125, f(nd::array{{1.5, 2.0, 7.0}, {-2.25, 7.0, 2.125}}));
  EXPECT_ARRAY_EQ(1.5 - 2.0, f(nd::array{{1.5, -2.0}}));

  f = nd::functional::reduction([] { return 0; },
                                [](const return_wrapper<int> &res, int x, int y) { res += max(x, y); });
  EXPECT_ARRAY_EQ(36, f(nd::array{{4, -1, 7}, {8, 0, 5}}, nd::array{{9, -3, -5}, {0, 2, 11}}));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceReduce_KeepDim) {
  nd::callable f =
      nd::functional::reduction([] { return 0.0; }, [](const return_wrapper<double> &res, double x) { res += x; });
  EXPECT_ARRAY_EQ(nd::array({{1.5 + 2.0 + 7.0 - 2.25 + 7.0 + 2.125}}),
                  f({{{1.5, 2.0, 7.0}, {-2.25, 7.0, 2.125}}}, {{"keepdims", true}}));

  f = nd::functional::reduction([] { return 0; },
                                [](const return_wrapper<int> &res, int x, int y) { res += max(x, y); });
  EXPECT_ARRAY_EQ(nd::array({{36}}),
                  f({nd::array{{4, -1, 7}, {8, 0, 5}}, nd::array{{9, -3, -5}, {0, 2, 11}}}, {{"keepdims", true}}));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_BroadcastReduce) {
  nd::callable f =
      nd::functional::reduction([] { return 0.0; }, [](const return_wrapper<double> &res, double x) { res += x; });
  EXPECT_ARRAY_EQ((nd::array{1.5 + 2.0 + 7.0, -2.25 + 7.0 + 2.125}),
                  f({nd::array{{1.5, 2.0, 7.0}, {-2.25, 7.0, 2.125}}}, {{"axes", {1}}}));
  EXPECT_ARRAY_EQ(nd::array{1.5 - 2.0}, f({nd::array{{1.5, -2.0}}}, {{"axes", {1}}}));

  f = nd::functional::reduction([] { return 0; },
                                [](const return_wrapper<int> &res, int x, int y) { res += max(x, y); });
  EXPECT_ARRAY_EQ(nd::array({15, 21}),
                  f({nd::array{{4, -1, 7}, {8, 0, 5}}, nd::array{{9, -3, -5}, {0, 2, 11}}}, {{"axes", {1}}}));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_BroadcastReduce_KeepDim) {
  nd::callable f =
      nd::functional::reduction([] { return 0.0; }, [](const return_wrapper<double> &res, double x) { res += x; });
  EXPECT_ARRAY_EQ(nd::array({{1.5 + 2.0 + 7.0}, {-2.25 + 7.0 + 2.125}}),
                  f({{{1.5, 2.0, 7.0}, {-2.25, 7.0, 2.125}}}, {{"axes", {1}}, {"keepdims", true}}));

  f = nd::functional::reduction([] { return 0; },
                                [](const return_wrapper<int> &res, int x, int y) { res += max(x, y); });
  EXPECT_ARRAY_EQ(nd::array({{15}, {21}}), f({nd::array{{4, -1, 7}, {8, 0, 5}}, nd::array{{9, -3, -5}, {0, 2, 11}}},
                                             {{"axes", {1}}, {"keepdims", true}}));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceBroadcast) {
  nd::callable f =
      nd::functional::reduction([] { return 0.0; }, [](const return_wrapper<double> &res, double x) { res += x; });
  EXPECT_ARRAY_EQ((nd::array{1.5 - 2.25, 2.0 + 7.0, 7.0 + 2.125}),
                  f({nd::array{{1.5, 2.0, 7.0}, {-2.25, 7.0, 2.125}}}, {{"axes", {0}}}));
  EXPECT_ARRAY_EQ(nd::array({1.5, -2.0}), f({nd::array{{1.5, -2.0}}}, {{"axes", {0}}}));

  f = nd::functional::reduction([] { return 0; },
                                [](const return_wrapper<int> &res, int x, int y) { res += max(x, y); });
  EXPECT_ARRAY_EQ(nd::array({17, 1, 18}),
                  f({nd::array{{4, -1, 7}, {8, 0, 5}}, nd::array{{9, -3, -5}, {0, 2, 11}}}, {{"axes", {0}}}));
}

TEST(Reduction, BuiltinSum_Lift2D_StridedStrided_ReduceBroadcast_KeepDim) {
  nd::callable f =
      nd::functional::reduction([] { return 0.0; }, [](const return_wrapper<double> &res, double x) { res += x; });
  EXPECT_ARRAY_EQ((nd::array{{1.5 - 2.25, 2.0 + 7.0, 7.0 + 2.125}}),
                  f({{{1.5, 2.0, 7.0}, {-2.25, 7.0, 2.125}}}, {{"axes", {0}}, {"keepdims", true}}));

  f = nd::functional::reduction([] { return 0; },
                                [](const return_wrapper<int> &res, int x, int y) { res += max(x, y); });
  EXPECT_ARRAY_EQ(nd::array({{17, 1, 18}}), f({nd::array{{4, -1, 7}, {8, 0, 5}}, nd::array{{9, -3, -5}, {0, 2, 11}}},
                                              {{"axes", {0}}, {"keepdims", true}}));
}

TEST(Reduction, FixedVar) {
  nd::callable f =
      nd::functional::reduction([] { return 0.0; }, [](const return_wrapper<double> &res, double x) { res += x; });

  nd::array a = nd::empty("3 * var * float64");
  a(0).vals() = {1, 2};
  a(1).vals() = {3, 4, 5};
  a(2).vals() = {6, 7, 8, 9};

  EXPECT_ARRAY_EQ(45.0, f(a));
}

TEST(Reduction, FixedVarWithAxes) {
  nd::callable f =
      nd::functional::reduction([] { return 0.0; }, [](const return_wrapper<double> &res, double x) { res += x; });

  nd::array a = nd::empty("3 * var * float64");
  a(0).vals() = {1, 2};
  a(1).vals() = {3, 4, 5};
  a(2).vals() = {6, 7, 8, 9};

  EXPECT_ARRAY_EQ((nd::array{3.0, 12.0, 30.0}), f({a}, {{"axes", nd::array{1}}}));
}

TEST(Reduction, BuiltinSum_Lift3D_StridedStridedStrided_ReduceReduceReduce) {
  nd::callable f =
      nd::functional::reduction([] { return 0.0; }, [](const return_wrapper<double> &res, double x) { res += x; });
  EXPECT_ARRAY_EQ(1.5 - 2.375 + 2.0 + 1.25 + 7.0 - 0.5 - 2.25 + 1.0 + 7.0 + 2.125 + 0.25,
                  f(initializer_list<initializer_list<initializer_list<double>>>{{{1.5, -2.375}, {2, 1.25}, {7, -0.5}},
                                                                                 {{-2.25, 1}, {7, 0}, {2.125, 0.25}}}));

  f = nd::functional::reduction([] { return 0; },
                                [](const return_wrapper<int> &res, int x, int y) { res += max(x, y); });
  EXPECT_ARRAY_EQ(62, f(nd::array{{{4, -1}, {7, 8}, {0, 5}}, {{-2, 1}, {3, -4}, {-9, 6}}},
                        nd::array{{{9, -3}, {-5, 0}, {2, 11}}, {{7, -14}, {10, 2}, {0, 5}}}));
}

TEST(Reduction, BuiltinSum_Lift3D_StridedStridedStrided_BroadcastReduceReduce) {
  nd::callable f =
      nd::functional::reduction([] { return 0.0; }, [](const return_wrapper<double> &res, double x) { res += x; });
  EXPECT_ARRAY_EQ((nd::array{1.5 - 2.375 + 2.0 + 1.25 + 7.0 - 0.5, -2.25 + 1.0 + 7.0 + 2.125 + 0.25}),
                  f({nd::array{{{1.5, -2.375}, {2.0, 1.25}, {7.0, -0.5}}, {{-2.25, 1.0}, {7.0, 0.0}, {2.125, 0.25}}}},
                    {{"axes", {1, 2}}}));

  f = nd::functional::reduction([] { return 0; },
                                [](const return_wrapper<int> &res, int x, int y) { res += max(x, y); });
  EXPECT_ARRAY_EQ(nd::array({36, 26}), f({nd::array{{{4, -1}, {7, 8}, {0, 5}}, {{-2, 1}, {3, -4}, {-9, 6}}},
                                          nd::array{{{9, -3}, {-5, 0}, {2, 11}}, {{7, -14}, {10, 2}, {0, 5}}}},
                                         {{"axes", {1, 2}}}));
}

TEST(Reduction, BuiltinSum_Lift3D_StridedStridedStrided_ReduceBroadcastReduce) {
  nd::callable f =
      nd::functional::reduction([] { return 0.0; }, [](const return_wrapper<double> &res, double x) { res += x; });
  EXPECT_ARRAY_EQ(
      (nd::array{1.5 - 2.375 - 2.25 + 1.0, 2.0 + 1.25 + 7.0, 7.0 - 0.5 + 2.125 + 0.25}),
      f({{{{1.5, -2.375}, {2.0, 1.25}, {7.0, -0.5}}, {{-2.25, 1.0}, {7.0, 0.0}, {2.125, 0.25}}}}, {{"axes", {0, 2}}}));

  f = nd::functional::reduction([] { return 0; },
                                [](const return_wrapper<int> &res, int x, int y) { res += max(x, y); });
  EXPECT_ARRAY_EQ(nd::array({16, 27, 19}), f({nd::array{{{4, -1}, {7, 8}, {0, 5}}, {{-2, 1}, {3, -4}, {-9, 6}}},
                                              nd::array{{{9, -3}, {-5, 0}, {2, 11}}, {{7, -14}, {10, 2}, {0, 5}}}},
                                             {{"axes", {0, 2}}}));
}

TEST(Reduction, Except) {
  // Cannot have a null child
  EXPECT_THROW(nd::functional::reduction([] { return 0; }, nd::callable()), invalid_argument);

  // Cannot have a null identity
  EXPECT_THROW(nd::functional::reduction(nd::callable(), [](const return_wrapper<int> &res, int x) { res += x; }),
               invalid_argument);

  // Cannot have a child with no arguments
  //  EXPECT_THROW(nd::functional::reduction(nd::functional::apply([]() { return 0; })), invalid_argument);
}
