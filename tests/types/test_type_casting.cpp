//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/types/fixed_string_type.hpp>
#include <dynd/array.hpp>

using namespace std;
using namespace dynd;

TEST(TypeCasting, IsLosslessAssignment) {
    // Boolean casting
    EXPECT_TRUE(is_lossless_assignment(ndt::type(bool_id), ndt::type(bool_id)));

    // Signed int casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int8_id),  ndt::type(bool_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int8_id),  ndt::type(int8_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int8_id),  ndt::type(int16_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int8_id),  ndt::type(int32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int8_id),  ndt::type(int64_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int16_id), ndt::type(bool_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int16_id), ndt::type(int8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int16_id), ndt::type(int16_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int16_id), ndt::type(int32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int16_id), ndt::type(int64_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int32_id), ndt::type(bool_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int32_id), ndt::type(int8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int32_id), ndt::type(int16_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int32_id), ndt::type(int32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int32_id), ndt::type(int64_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_id), ndt::type(bool_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_id), ndt::type(int8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_id), ndt::type(int16_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_id), ndt::type(int32_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_id), ndt::type(int64_id)));

    // Unsigned int casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint8_id),  ndt::type(uint8_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint8_id),  ndt::type(uint16_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint8_id),  ndt::type(uint32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint8_id),  ndt::type(uint64_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint16_id), ndt::type(uint8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint16_id), ndt::type(uint16_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint16_id), ndt::type(uint32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint16_id), ndt::type(uint64_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint32_id), ndt::type(uint8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint32_id), ndt::type(uint16_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint32_id), ndt::type(uint32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint32_id), ndt::type(uint64_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint64_id), ndt::type(uint8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint64_id), ndt::type(uint16_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint64_id), ndt::type(uint32_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint64_id), ndt::type(uint64_id)));

    // Float casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float32_id), ndt::type(float32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float32_id), ndt::type(float64_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_id), ndt::type(float32_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_id), ndt::type(float64_id)));

    // Complex Casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float32_id), ndt::type(complex_float32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float32_id), ndt::type(complex_float64_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_id), ndt::type(complex_float32_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_id), ndt::type(complex_float64_id)));

    // String casting
    // String conversions report false, so that assignments encodings
    // get validated on assignment
    EXPECT_FALSE(is_lossless_assignment(ndt::make_type<ndt::fixed_string_type>(16, string_encoding_utf_16),
                            ndt::make_type<ndt::fixed_string_type>(16, string_encoding_utf_16)));
    EXPECT_FALSE(is_lossless_assignment(ndt::make_type<ndt::fixed_string_type>(16, string_encoding_utf_8),
                            ndt::make_type<ndt::fixed_string_type>(12, string_encoding_utf_8)));
    EXPECT_FALSE(is_lossless_assignment(ndt::make_type<ndt::fixed_string_type>(12, string_encoding_utf_32),
                            ndt::make_type<ndt::fixed_string_type>(16, string_encoding_utf_32)));
    EXPECT_FALSE(is_lossless_assignment(ndt::make_type<ndt::fixed_string_type>(16, string_encoding_utf_16),
                            ndt::make_type<ndt::fixed_string_type>(16, string_encoding_utf_32)));
    EXPECT_FALSE(is_lossless_assignment(ndt::make_type<ndt::fixed_string_type>(16, string_encoding_utf_8),
                            ndt::make_type<ndt::fixed_string_type>(16, string_encoding_utf_16)));

    // Int -> UInt casting
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint64_id), ndt::type(int8_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint64_id), ndt::type(int64_id)));

    // UInt -> Int casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_id), ndt::type(uint8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_id), ndt::type(uint16_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_id), ndt::type(uint32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int64_id), ndt::type(uint64_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int32_id), ndt::type(uint8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int32_id), ndt::type(uint16_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int32_id), ndt::type(uint32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int32_id), ndt::type(int64_id)));

    // Int -> Float casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float32_id), ndt::type(int8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float32_id), ndt::type(int16_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float32_id), ndt::type(int32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float32_id), ndt::type(int64_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_id), ndt::type(int8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_id), ndt::type(int16_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_id), ndt::type(int32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float64_id), ndt::type(int64_id)));

    // Int -> Complex casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float32_id), ndt::type(int8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float32_id), ndt::type(int16_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float32_id), ndt::type(int32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float32_id), ndt::type(int64_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_id), ndt::type(int8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_id), ndt::type(int16_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_id), ndt::type(int32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float64_id), ndt::type(int64_id)));

    // UInt -> Float casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float32_id), ndt::type(uint8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float32_id), ndt::type(uint16_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float32_id), ndt::type(uint32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float32_id), ndt::type(uint64_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_id), ndt::type(uint8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_id), ndt::type(uint16_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_id), ndt::type(uint32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float64_id), ndt::type(uint64_id)));

    // UInt -> Complex casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float32_id), ndt::type(uint8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float32_id), ndt::type(uint16_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float32_id), ndt::type(uint32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float32_id), ndt::type(uint64_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_id), ndt::type(uint8_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_id), ndt::type(uint16_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_id), ndt::type(uint32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float64_id), ndt::type(uint64_id)));

    // Float -> Int casting
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int8_id),  ndt::type(float32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int16_id), ndt::type(float32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int32_id), ndt::type(float32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int64_id), ndt::type(float32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int8_id),  ndt::type(float64_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int16_id), ndt::type(float64_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int32_id), ndt::type(float64_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int64_id), ndt::type(float64_id)));

    // Float -> UInt casting
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint8_id),  ndt::type(float32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint16_id), ndt::type(float32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint32_id), ndt::type(float32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint64_id), ndt::type(float32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint8_id),  ndt::type(float64_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint16_id), ndt::type(float64_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint32_id), ndt::type(float64_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint64_id), ndt::type(float64_id)));

    // Float -> Complex casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float32_id), ndt::type(float32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float32_id), ndt::type(float64_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_id), ndt::type(float32_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_id), ndt::type(float64_id)));

    // Complex -> Float casting
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float32_id), ndt::type(complex_float32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float32_id), ndt::type(complex_float64_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float64_id), ndt::type(complex_float32_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float64_id), ndt::type(complex_float64_id)));
}


TEST(TypeCasting, StringToInt32) {
    nd::array a = nd::empty(ndt::make_type<int>());

    // Test the limits of string to int conversion
    a.vals() = "2147483647";
    EXPECT_EQ(2147483647, a.as<int>());
    a.vals() = "-2147483648";
    EXPECT_EQ(-2147483647 - 1, a.as<int>());
    EXPECT_THROW(a.vals() = "2147483648", overflow_error);
    EXPECT_THROW(a.vals() = "-2147483649", overflow_error);

    // Trailing ".0" is permitted
    a.vals() = "1234.";
    EXPECT_EQ(1234, a.as<int>());
    a.vals() = "2345.0";
    EXPECT_EQ(2345, a.as<int>());
    a.vals() = "3456.00000";
    EXPECT_EQ(3456, a.as<int>());

    // Simple "1e5" positive exponent cases are permitted
    a.vals() = "1e5";
    EXPECT_EQ(100000, a.as<int>());
    a.vals() = "1e9";
    EXPECT_EQ(1000000000, a.as<int>());
    a.vals() = "2e9";
    EXPECT_EQ(2000000000, a.as<int>());
    a.vals() = "-21e8";
    EXPECT_EQ(-2100000000, a.as<int>());
    EXPECT_THROW(a.vals() = "3e9", overflow_error);
}

/*
ToDo: Reenable this.

TEST(TypeCasting, StringToInt64) {
    nd::array a = nd::empty(ndt::make_type<int64_t>());

    // Test the limits of string to int conversion
    a.vals() = "-0";
    EXPECT_EQ(0LL, a.as<int64_t>());
    a.vals() = "9223372036854775807";
    EXPECT_EQ(9223372036854775807LL, a.as<int64_t>());
    a.vals() = "-9223372036854775808";
    EXPECT_EQ(-9223372036854775807LL - 1LL, a.as<int64_t>());
    EXPECT_THROW(a.vals() = "9223372036854775808", overflow_error);
    EXPECT_THROW(a.vals() = "-9223372036854775809", overflow_error);

    // Simple "1e5" positive exponent cases are permitted
    a.vals() = "1e18";
    EXPECT_EQ(1000000000000000000LL, a.as<int64_t>());
    a.vals() = "922e16";
    EXPECT_EQ(9220000000000000000LL, a.as<int64_t>());
    EXPECT_THROW(a.vals() = "1e19", overflow_error);
}
*/

/*
TEST(TypeCasting, StringToInt128) {
    nd::array a = nd::empty(ndt::make_type<int128>());

    // Test the limits of string to int conversion
    a.vals() = "-0";
    EXPECT_EQ(0LL, a.as<int64_t>());
    a.vals() = "-170141183460469231731687303715884105728";
    EXPECT_EQ(0x8000000000000000ULL, a.as<int128>().m_hi);
    EXPECT_EQ(0ULL, a.as<int128>().m_lo);
    a.vals() = "170141183460469231731687303715884105727";
    EXPECT_EQ(0x7fffffffffffffffULL, a.as<int128>().m_hi);
    EXPECT_EQ(0xffffffffffffffffULL, a.as<int128>().m_lo);
    EXPECT_THROW(a.vals() = "170141183460469231731687303715884105728",
                 overflow_error);
    EXPECT_THROW(a.vals() = "-170141183460469231731687303715884105729",
                 overflow_error);

    // Simple "1e5" positive exponent cases are permitted
    a.vals() = "1e18";
    EXPECT_EQ(1000000000000000000LL, a.as<int64_t>());
    a.vals() = "922e26";
    EXPECT_EQ(0x129ea0d6fULL, a.as<int128>().m_hi);
    EXPECT_EQ(0x287e2f8928000000ULL, a.as<int128>().m_lo);
    EXPECT_THROW(a.vals() = "1e40", overflow_error);
}
*/

/*
TEST(TypeCasting, StringToUInt64) {
    nd::array a = nd::empty(ndt::make_type<uint64_t>());

    // Test the limits of string to int conversion
    a.vals() = "0";
    EXPECT_EQ(0u, a.as<uint64_t>());
    a.vals() = "18446744073709551615";
    EXPECT_EQ(18446744073709551615ULL, a.as<uint64_t>());
    EXPECT_THROW(a.vals() = "18446744073709551616", out_of_range);
    EXPECT_THROW(a.vals() = "-1", invalid_argument);

    // Simple "1e5" positive exponent cases are permitted
    a.vals() = "1e19";
    EXPECT_EQ(10000000000000000000ULL, a.as<uint64_t>());
    a.vals() = "1844e15";
    EXPECT_EQ(1844000000000000000ULL, a.as<uint64_t>());
    EXPECT_THROW(a.vals() = "1845e20", out_of_range);
    EXPECT_THROW(a.vals() = "1e20", out_of_range);
}
*/
