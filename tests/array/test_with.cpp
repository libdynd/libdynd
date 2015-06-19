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

template <type_id_t type_id>
struct r2c;

template <>
struct r2c<float32_type_id> {
  static const type_id_t value = complex_float32_type_id;
};

template <>
struct r2c<float64_type_id> {
  static const type_id_t value = complex_float64_type_id;
};

template <type_id_t TypeID0, type_id_t TypeID1>
struct S;

template <type_id_t TypeID>
struct S<TypeID, r2c<TypeID>::value> {
};

TEST(With1DStrided, ViewData)
{
  nd::array a = {1, 3, 5, 7};
  // Contiguous stride
  nd::with_1d_stride<int>(
      a, [&](intptr_t size, intptr_t stride, const int *data) {
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
  nd::with_1d_stride<int>(
      a(irange().by(2)), [&](intptr_t size, intptr_t stride, const int *data) {
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
  nd::with_1d_stride<int>(
      a, [&](intptr_t size, intptr_t stride, const int *data) {
        ASSERT_EQ(4, size);
        EXPECT_EQ(1, stride);
        EXPECT_EQ(1, data[0 * stride]);
        EXPECT_EQ(3, data[1 * stride]);
        EXPECT_EQ(5, data[2 * stride]);
        EXPECT_EQ(7, data[3 * stride]);
        EXPECT_NE(a.get_data(), reinterpret_cast<const char *>(data));
      });
  // Every second element
  nd::with_1d_stride<int>(
      a(irange().by(2)), [&](intptr_t size, intptr_t stride, const int *data) {
        ASSERT_EQ(2, size);
        EXPECT_EQ(1, stride);
        EXPECT_EQ(1, data[0 * stride]);
        EXPECT_EQ(5, data[1 * stride]);
        EXPECT_NE(a.get_data(), reinterpret_cast<const char *>(data));
      });
}
