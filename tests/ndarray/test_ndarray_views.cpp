//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <inc_gtest.hpp>

#include <dnd/ndarray.hpp>
#include <dnd/ndarray_arange.hpp>
#include <dnd/dtypes/dtype_alignment.hpp>
#include <dnd/dtypes/convert_dtype.hpp>
#include <dnd/dtypes/view_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(NDArrayViews, OneDimensionalRawMemory) {
    ndarray a, b;
    char c_values[8];
    uint64_t u8_value;

    // Make an 8 byte aligned array of 80 chars
    a = ndarray(10, make_dtype<uint64_t>());
    a = a.view_as_dtype(make_dtype<char>());

    // Initialize the char values from a uint64_t,
    // to avoid having to know the endianness
    u8_value = 0x102030405060708ULL;
    memcpy(c_values, &u8_value, 8);
    a(irange() < 8).vals() = c_values;
    b = a.view_as_dtype<uint64_t>();
    EXPECT_EQ(make_dtype<uint64_t>(), b.get_dtype());
    EXPECT_EQ(1, b.get_ndim());
    EXPECT_EQ(10, b.get_shape()[0]);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(u8_value, b(0).as<uint64_t>());
    b(0).vals() = 0x0505050505050505ULL;
    EXPECT_EQ(5, a(0).as<char>());

    // The system should automatically apply unaligned<>
    // where necessary
    a(1 <= irange() < 9).vals() = c_values;
    b = a(1 <= irange() < 73).view_as_dtype<uint64_t>();
    EXPECT_EQ(make_view_dtype(make_dtype<uint64_t>(), make_fixedbytes_dtype(8, 1)), b.get_dtype());
    EXPECT_EQ(1, b.get_ndim());
    EXPECT_EQ(9, b.get_shape()[0]);
    EXPECT_EQ(a.get_readonly_originptr() + 1, b.get_readonly_originptr());
    EXPECT_EQ(u8_value, b(0).as<uint64_t>());
}

TEST(NDArrayViews, MultiDimensionalRawMemory) {
    ndarray a, b;
    uint32_t values[2][3] = {{1,2,3}, {0xffffffff, 0x80000000, 0}};

    a = values;

    // Should throw if the view dtype is the wrong size
    EXPECT_THROW(b = a.view_as_dtype<int16_t>(), runtime_error);

    b = a.view_as_dtype<int32_t>();
    EXPECT_EQ(make_dtype<int32_t>(), b.get_dtype());
    EXPECT_EQ(2, b.get_ndim());
    EXPECT_EQ(2, b.get_shape()[0]);
    EXPECT_EQ(3, b.get_shape()[1]);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(1, b(0, 0).as<int32_t>());
    EXPECT_EQ(2, b(0, 1).as<int32_t>());
    EXPECT_EQ(3, b(0, 2).as<int32_t>());
    EXPECT_EQ(-1, b(1, 0).as<int32_t>());
    EXPECT_EQ(std::numeric_limits<int32_t>::min(), b(1, 1).as<int32_t>());
    EXPECT_EQ(0, b(1, 2).as<int32_t>());
}

TEST(NDArrayViews, ExpressionDType) {
    ndarray a, a_u2, b;
    uint32_t values[2][3] = {{1,2,3}, {0xffff, 0x8000, 0}};

    // Create a conversion from uint32_t -> uint16_t, followed by a
    // view uint16_t -> int16_t
    a = values;
    a_u2 = a.as_dtype<uint16_t>();
    EXPECT_EQ((make_convert_dtype<uint16_t, uint32_t>()), a_u2.get_dtype());

    // Wrong size, so should throw
    EXPECT_THROW(b = a_u2.view_as_dtype<int32_t>(), runtime_error);

    b = a_u2.view_as_dtype<int16_t>();
    EXPECT_EQ((make_view_dtype(make_dtype<int16_t>(), make_convert_dtype<uint16_t, uint32_t>())), b.get_dtype());
    EXPECT_EQ(2, b.get_ndim());
    EXPECT_EQ(2, b.get_shape()[0]);
    EXPECT_EQ(3, b.get_shape()[1]);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(1, b(0, 0).as<int16_t>());
    EXPECT_EQ(2, b(0, 1).as<int16_t>());
    EXPECT_EQ(3, b(0, 2).as<int16_t>());
    EXPECT_EQ(-1, b(1, 0).as<int16_t>());
    EXPECT_EQ(std::numeric_limits<int16_t>::min(), b(1, 1).as<int16_t>());
    EXPECT_EQ(0, b(1, 2).as<int16_t>());
}
