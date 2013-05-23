//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <inc_gtest.hpp>

#include <dynd/ndobject.hpp>
#include <dynd/ndobject_arange.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/dtypes/fixedbytes_dtype.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(NDObjectViews, OneDimensionalRawMemory) {
    ndobject a, b;
    signed char c_values[8];
    uint64_t u8_value;

    // Make an 8 byte aligned array of 80 chars
    a = make_strided_ndobject(10, make_dtype<uint64_t>());
    a = a.view_scalars(make_dtype<char>());

    // Initialize the char values from a uint64_t,
    // to avoid having to know the endianness
    u8_value = 0x102030405060708ULL;
    memcpy(c_values, &u8_value, 8);
    a.at(irange() < 8).vals() = c_values;
    b = a.view_scalars<uint64_t>();
    EXPECT_EQ(make_strided_dim_dtype(make_dtype<uint64_t>()), b.get_dtype());
    EXPECT_EQ(1u, b.get_shape().size());
    EXPECT_EQ(10, b.get_shape()[0]);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(u8_value, b.at(0).as<uint64_t>());
    b.at(0).vals() = 0x0505050505050505ULL;
    EXPECT_EQ(5, a.at(0).as<char>());

    // The system should automatically apply unaligned<>
    // where necessary
    a.at(1 <= irange() < 9).vals() = c_values;
    b = a.at(1 <= irange() < 73).view_scalars<uint64_t>();
    EXPECT_EQ(make_strided_dim_dtype(make_view_dtype(make_dtype<uint64_t>(), make_fixedbytes_dtype(8, 1))),
                    b.get_dtype());
    EXPECT_EQ(1u, b.get_shape().size());
    EXPECT_EQ(9, b.get_shape()[0]);
    EXPECT_EQ(a.get_readonly_originptr() + 1, b.get_readonly_originptr());
    EXPECT_EQ(u8_value, b.at(0).as<uint64_t>());
}

TEST(NDObjectViews, MultiDimensionalRawMemory) {
    ndobject a, b;
    uint32_t values[2][3] = {{1,2,3}, {0xffffffff, 0x80000000, 0}};

    a = values;

    // Should throw if the view dtype is the wrong size
    EXPECT_THROW(b = a.view_scalars<int16_t>(), runtime_error);

    b = a.view_scalars<int32_t>();
    EXPECT_EQ(make_strided_dim_dtype(make_dtype<int32_t>(), 2), b.get_dtype());
    EXPECT_EQ(2u, b.get_shape().size());
    EXPECT_EQ(2, b.get_shape()[0]);
    EXPECT_EQ(3, b.get_shape()[1]);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(1, b.at(0, 0).as<int32_t>());
    EXPECT_EQ(2, b.at(0, 1).as<int32_t>());
    EXPECT_EQ(3, b.at(0, 2).as<int32_t>());
    EXPECT_EQ(-1, b.at(1, 0).as<int32_t>());
    EXPECT_EQ(std::numeric_limits<int32_t>::min(), b.at(1, 1).as<int32_t>());
    EXPECT_EQ(0, b.at(1, 2).as<int32_t>());
}

TEST(NDObjectViews, ExpressionDType) {
    ndobject a, a_u2, b;
    uint32_t values[2][3] = {{1,2,3}, {0xffff, 0x8000, 0}};

    // Create a conversion from uint32_t -> uint16_t, followed by a
    // view uint16_t -> int16_t
    a = values;
    a_u2 = a.ucast<uint16_t>();
    EXPECT_EQ(make_strided_dim_dtype(make_convert_dtype<uint16_t, uint32_t>(), 2), a_u2.get_dtype());

    // Wrong size, so should throw
    EXPECT_THROW(b = a_u2.view_scalars<int32_t>(), runtime_error);

    b = a_u2.view_scalars<int16_t>();
    EXPECT_EQ(make_strided_dim_dtype(make_view_dtype(make_dtype<int16_t>(), make_convert_dtype<uint16_t, uint32_t>()), 2),
                    b.get_dtype());
    EXPECT_EQ(2u, b.get_shape().size());
    EXPECT_EQ(2, b.get_shape()[0]);
    EXPECT_EQ(3, b.get_shape()[1]);
    EXPECT_EQ(a.get_readonly_originptr(), b.get_readonly_originptr());
    EXPECT_EQ(1, b.at(0, 0).as<int16_t>());
    EXPECT_EQ(2, b.at(0, 1).as<int16_t>());
    EXPECT_EQ(3, b.at(0, 2).as<int16_t>());
    EXPECT_EQ(-1, b.at(1, 0).as<int16_t>());
    EXPECT_EQ(std::numeric_limits<int16_t>::min(), b.at(1, 1).as<int16_t>());
    EXPECT_EQ(0, b.at(1, 2).as<int16_t>());
}
