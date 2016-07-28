//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/index.hpp>
#include <dynd_assertions.hpp>

using namespace std;
using namespace dynd;

TEST(Callable, Take) {
  nd::array a, b, c;

  int avals[5] = {1, 2, 3, 4, 5};
  a = avals;

  // Masked take
  bool1 bvals[5] = {bool1(false), bool1(true), bool1(false), bool1(true), bool1(true)};
  b = bvals;
  c = nd::take(a, b);
  EXPECT_EQ(ndt::type("var * int"), c.get_type());
  ASSERT_EQ(3, c.get_dim_size());
  EXPECT_EQ(2, c(0).as<int>());
  EXPECT_EQ(4, c(1).as<int>());
  EXPECT_EQ(5, c(2).as<int>());

  // Indexed take
  intptr_t bvals2[4] = {3, 0, -1, 4};
  b = bvals2;
  c = nd::take(a, b);
  /*
    EXPECT_EQ(ndt::type("4 * int"), c.get_type());
    ASSERT_EQ(4, c.get_dim_size());
    EXPECT_EQ(4, c(0).as<int>());
    EXPECT_EQ(1, c(1).as<int>());
    EXPECT_EQ(5, c(2).as<int>());
    EXPECT_EQ(5, c(3).as<int>());
  */
}

TEST(Callable, TakeOfArray) {
  nd::array a, b, c;

  int avals[3][2] = {{0, 1}, {2, 3}, {4, 5}};
  a = avals;

  // Masked take
  bool1 bvals[3] = {bool1(true), bool1(false), bool1(true)};
  b = bvals;
  c = nd::take(a, b);
  EXPECT_EQ(ndt::type("var * 2 * int"), c.get_type());
  ASSERT_EQ(2, c.get_dim_size());
  ASSERT_EQ(2, c.get_shape()[1]);
  EXPECT_EQ(0, c(0, 0).as<int>());
  EXPECT_EQ(1, c(0, 1).as<int>());
  EXPECT_EQ(4, c(1, 0).as<int>());
  EXPECT_EQ(5, c(1, 1).as<int>());

  /*
    // Indexed take
    intptr_t bvals2[4] = {1, 0, -1, -2};
    b = bvals2;
    c = nd::take(a, b);
    EXPECT_EQ(ndt::type("4 * 2 * int"), c.get_type());
    ASSERT_EQ(4, c.get_dim_size());
    ASSERT_EQ(2, c.get_shape()[1]);
    EXPECT_EQ(2, c(0, 0).as<int>());
    EXPECT_EQ(3, c(0, 1).as<int>());
    EXPECT_EQ(0, c(1, 0).as<int>());
    EXPECT_EQ(1, c(1, 1).as<int>());
    EXPECT_EQ(4, c(2, 0).as<int>());
    EXPECT_EQ(5, c(2, 1).as<int>());
    EXPECT_EQ(2, c(3, 0).as<int>());
    EXPECT_EQ(3, c(3, 1).as<int>());
  */
}
