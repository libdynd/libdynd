//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "../dynd_assertions.hpp"
#include "../test_memory.hpp"

#include <dynd/array_range.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/with.hpp>

using namespace std;
using namespace dynd;

TEST(With1DStrided, ViewData)
{
  nd::array a = {1, 3, 5, 7};
  // Contiguous stride
  nd::with_1d_stride<int>(a, [&](intptr_t size, intptr_t stride, const int *data) {
    ASSERT_EQ(4, size);
    EXPECT_EQ(1, stride);
    EXPECT_EQ(1, data[0 * stride]);
    EXPECT_EQ(3, data[1 * stride]);
    EXPECT_EQ(5, data[2 * stride]);
    EXPECT_EQ(7, data[3 * stride]);
    // This should have resulted in a view
    EXPECT_EQ(a.get_data(), reinterpret_cast<const char *>(data));
  });
  // Every second element
  nd::with_1d_stride<int>(a(irange().by(2)), [&](intptr_t size, intptr_t stride, const int *data) {
    ASSERT_EQ(2, size);
    EXPECT_EQ(2, stride);
    EXPECT_EQ(1, data[0 * stride]);
    EXPECT_EQ(5, data[1 * stride]);
    // This should have resulted in a view
    EXPECT_EQ(a.get_data(), reinterpret_cast<const char *>(data));
  });
}

TEST(With1DStrided, ConvertData)
{
  nd::array a = {1.f, 3.f, 5.f, 7.f};
  // Contiguous stride
  nd::with_1d_stride<int>(a, [&](intptr_t size, intptr_t stride, const int *data) {
    ASSERT_EQ(4, size);
    EXPECT_EQ(1, stride);
    EXPECT_EQ(1, data[0 * stride]);
    EXPECT_EQ(3, data[1 * stride]);
    EXPECT_EQ(5, data[2 * stride]);
    EXPECT_EQ(7, data[3 * stride]);
    EXPECT_NE(a.get_data(), reinterpret_cast<const char *>(data));
  });
  // Every second element
  nd::with_1d_stride<int>(a(irange().by(2)), [&](intptr_t size, intptr_t stride, const int *data) {
    ASSERT_EQ(2, size);
    EXPECT_EQ(1, stride);
    EXPECT_EQ(1, data[0 * stride]);
    EXPECT_EQ(5, data[1 * stride]);
    EXPECT_NE(a.get_data(), reinterpret_cast<const char *>(data));
  });
}

TEST(View, FixedDim)
{
  nd::array a = {0, 1, 2, 3, 4};

  auto vals = a.view<fixed_dim<int>>();
  EXPECT_TRUE((std::is_same<fixed_dim<int>, decltype(vals)>::value));
  EXPECT_EQ(0, vals(0));
  EXPECT_EQ(1, vals(1));
  EXPECT_EQ(2, vals(2));
  EXPECT_EQ(3, vals(3));
  EXPECT_EQ(4, vals(4));

  int i = 0;
  for (auto val : vals) {
    EXPECT_EQ(val, vals(i));
    ++i;
  }
}

TEST(View, FixedDimFixedDim)
{
  nd::array a = {{0, 1}, {2, 3}};

  auto vals = a.view<fixed_dim<fixed_dim<int>>>();
  EXPECT_TRUE((std::is_same<fixed_dim<fixed_dim<int>>, decltype(vals)>::value));
  EXPECT_EQ(0, vals(0, 0));
  EXPECT_EQ(1, vals(0, 1));
  EXPECT_EQ(2, vals(1, 0));
  EXPECT_EQ(3, vals(1, 1));

  EXPECT_TRUE((std::is_same<fixed_dim<int>, decltype(vals(std::declval<intptr_t>()))>::value));
  EXPECT_EQ(0, vals(0)(0));
  EXPECT_EQ(1, vals(0)(1));
  EXPECT_EQ(2, vals(1)(0));
  EXPECT_EQ(3, vals(1)(1));

  int i = 0;
  for (auto val : vals) {
    EXPECT_EQ(vals(i)(0), val(0));
    EXPECT_EQ(vals(i)(1), val(1));
    ++i;
  }
}

TEST(View, FixedDimFixedDimFixedDim)
{
  nd::array a = {{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};

  auto vals = a.view<fixed_dim<fixed_dim<fixed_dim<int>>>>();
  EXPECT_EQ(0, vals(0, 0, 0));
  EXPECT_EQ(1, vals(0, 0, 1));
  EXPECT_EQ(2, vals(0, 1, 0));
  EXPECT_EQ(3, vals(0, 1, 1));
  EXPECT_EQ(4, vals(1, 0, 0));
  EXPECT_EQ(5, vals(1, 0, 1));
  EXPECT_EQ(6, vals(1, 1, 0));
  EXPECT_EQ(7, vals(1, 1, 1));

  EXPECT_TRUE(
      (std::is_same<fixed_dim<int>, decltype(vals(std::declval<intptr_t>(), std::declval<intptr_t>()))>::value));
  EXPECT_EQ(0, vals(0, 0)(0));
  EXPECT_EQ(1, vals(0, 0)(1));
  EXPECT_EQ(2, vals(0, 1)(0));
  EXPECT_EQ(3, vals(0, 1)(1));
  EXPECT_EQ(4, vals(1, 0)(0));
  EXPECT_EQ(5, vals(1, 0)(1));
  EXPECT_EQ(6, vals(1, 1)(0));
  EXPECT_EQ(7, vals(1, 1)(1));

  EXPECT_TRUE((std::is_same<fixed_dim<fixed_dim<int>>, decltype(vals(std::declval<intptr_t>()))>::value));
  EXPECT_EQ(0, vals(0)(0, 0));
  EXPECT_EQ(1, vals(0)(0, 1));
  EXPECT_EQ(2, vals(0)(1, 0));
  EXPECT_EQ(3, vals(0)(1, 1));
  EXPECT_EQ(4, vals(1)(0, 0));
  EXPECT_EQ(5, vals(1)(0, 1));
  EXPECT_EQ(6, vals(1)(1, 0));
  EXPECT_EQ(7, vals(1)(1, 1));

  int i = 0;
  for (auto val : vals) {
    EXPECT_EQ(vals(i)(0, 0), val(0, 0));
    EXPECT_EQ(vals(i)(0, 1), val(0, 1));
    EXPECT_EQ(vals(i)(1, 0), val(1, 0));
    EXPECT_EQ(vals(i)(1, 1), val(1, 1));
    ++i;
  }
}
