//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/gtest.hpp>
#include <dynd/iterator.hpp>

using namespace std;
using namespace dynd;

TEST(Iterator, CArray1D) {
  int vals[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  auto it = dynd::begin(vals);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(vals[i], *it++);
  }
  EXPECT_TRUE(it == dynd::end(vals));
}

TEST(ConstIterator, CArray1D) {
  const int vals[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  auto it = dynd::begin(vals);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(vals[i], *it++);
  }
  EXPECT_TRUE(it == dynd::end(vals));
}

TEST(Iterator, CArray2D) {
  int vals[3][2] = {{0, 1}, {2, 3}, {4, 5}};

  auto it = dynd::begin<2>(vals);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(vals[i][j], *it++);
    }
  }
  EXPECT_TRUE(it == dynd::end<2>(vals));
}

TEST(ConstIterator, CArray2D) {
  const int vals[3][2] = {{0, 1}, {2, 3}, {4, 5}};

  auto it = dynd::begin<2>(vals);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(vals[i][j], *it++);
    }
  }
  EXPECT_TRUE(it == dynd::end<2>(vals));
}

TEST(Iterator, CArray3D) {
  int vals[4][3][2] = {
      {{0, 1}, {2, 3}, {4, 5}},
      {{6, 7}, {8, 9}, {10, 11}},
      {{12, 13}, {14, 15}, {16, 17}},
      {{18, 19}, {20, 21}, {22, 23}},
  };

  auto it = dynd::begin<3>(vals);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 2; ++k) {
        EXPECT_EQ(vals[i][j][k], *it++);
      }
    }
  }
  EXPECT_TRUE(it == dynd::end<3>(vals));
}

TEST(ConstIterator, CArray3D) {
  const int vals[4][3][2] = {
      {{0, 1}, {2, 3}, {4, 5}},
      {{6, 7}, {8, 9}, {10, 11}},
      {{12, 13}, {14, 15}, {16, 17}},
      {{18, 19}, {20, 21}, {22, 23}},
  };

  auto it = dynd::begin<3>(vals);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 2; ++k) {
        EXPECT_EQ(vals[i][j][k], *it++);
      }
    }
  }
  EXPECT_TRUE(it == dynd::end<3>(vals));
}
