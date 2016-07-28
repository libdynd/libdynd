//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/functional.hpp>
#include <dynd_assertions.hpp>

using namespace std;
using namespace dynd;

TEST(Outer, 1D) {
  nd::callable f = nd::functional::outer([](int x) { return x; });
  EXPECT_ARRAY_EQ(nd::array({0, 1}), f(nd::array{0, 1}));
}

TEST(Outer, 2D) {
  nd::callable f = nd::functional::outer([](int x, int y) { return x + y; });
  EXPECT_ARRAY_EQ(nd::array({{1, 3}, {3, 5}}), f(nd::array{0, 2}, nd::array{1, 3}));
  EXPECT_ARRAY_EQ(nd::array({{1, 3, 5}, {3, 5, 7}, {5, 7, 9}}), f(nd::array{0, 2, 4}, nd::array{1, 3, 5}));
  EXPECT_ARRAY_EQ(nd::array({1, 3}), f(nd::array{0, 2}, 1));
  EXPECT_ARRAY_EQ(nd::array({2, 4}), f(1, nd::array{1, 3}));
  EXPECT_ARRAY_EQ(1, f(0, 1));

  f = nd::functional::outer([](int x, int y) { return x - y; });
  EXPECT_ARRAY_EQ(nd::array({{-1, -3}, {1, -1}}), f(nd::array{0, 2}, nd::array{1, 3}));
  EXPECT_ARRAY_EQ(nd::array({{-1, -3, -5}, {1, -1, -3}, {3, 1, -1}}), f(nd::array{0, 2, 4}, nd::array{1, 3, 5}));
}

TEST(Outer, 3D) {
  nd::callable f = nd::functional::outer([](int x, int y, int z) { return x + y + z; });
  EXPECT_ARRAY_EQ(nd::array({{{3, 6}, {6, 9}}, {{6, 9}, {9, 12}}}),
                  f(nd::array{0, 3}, nd::array{1, 4}, nd::array{2, 5}));
  EXPECT_ARRAY_EQ(nd::array({3, 4}), f(0, 1, nd::array{2, 3}));
  EXPECT_ARRAY_EQ(3, f(0, 1, 2));
}
