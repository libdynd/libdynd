//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include <dynd/asarray.hpp>
#include <dynd_assertions.hpp>

using namespace std;
using namespace dynd;

/*
TEST(AsArray, PassThrough) {
  // When the array type matches the requested pattern, nd::asarray passes the
  // input through. These tests all compare the raw reference pointer of the
  // nd::array to confirm this is the case.
  nd::array a;

  a = 100;
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("int32")).get());
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("SInt")).get());
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("Int")).get());
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("... * Int")).get());
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("Any")).get());

  a = {3.15, 2.2, 7.7};
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("3 * float64")).get());
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("3 * Real")).get());
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("Fixed * float64")).get());
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("Fixed * Real")).get());
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("... * float64")).get());
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("... * Real")).get());
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("Any")).get());

  a = {true, false, false, true};
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("4 * bool")).get());
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("4 * Bool")).get());
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("Fixed * bool")).get());
  EXPECT_EQ(a.get(), nd::asarray(a, ndt::type("Fixed * Bool")).get());
}
*/
