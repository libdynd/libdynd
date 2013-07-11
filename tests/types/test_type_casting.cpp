//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/typed_data_assign.hpp>
#include <dynd/types/fixedstring_type.hpp>

using namespace std;
using namespace dynd;

TEST(DTypeCasting, IsLosslessAssignment) {
    // Boolean casting
    EXPECT_TRUE(is_lossless_assignment(ndt::type(bool_type_id), ndt::type(bool_type_id)));

    // Signed int casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int8_type_id),  ndt::type(bool_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int8_type_id),  ndt::type(int8_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int8_type_id),  ndt::type(int16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int8_type_id),  ndt::type(int32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int8_type_id),  ndt::type(int64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int16_type_id), ndt::type(bool_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int16_type_id), ndt::type(int8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int16_type_id), ndt::type(int16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int16_type_id), ndt::type(int32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int16_type_id), ndt::type(int64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int32_type_id), ndt::type(bool_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int32_type_id), ndt::type(int8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int32_type_id), ndt::type(int16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int32_type_id), ndt::type(int32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int32_type_id), ndt::type(int64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_type_id), ndt::type(bool_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_type_id), ndt::type(int8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_type_id), ndt::type(int16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_type_id), ndt::type(int32_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_type_id), ndt::type(int64_type_id)));

    // Unsigned int casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint8_type_id),  ndt::type(uint8_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint8_type_id),  ndt::type(uint16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint8_type_id),  ndt::type(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint8_type_id),  ndt::type(uint64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint16_type_id), ndt::type(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint16_type_id), ndt::type(uint16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint16_type_id), ndt::type(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint16_type_id), ndt::type(uint64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint32_type_id), ndt::type(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint32_type_id), ndt::type(uint16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint32_type_id), ndt::type(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint32_type_id), ndt::type(uint64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint64_type_id), ndt::type(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint64_type_id), ndt::type(uint16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint64_type_id), ndt::type(uint32_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(uint64_type_id), ndt::type(uint64_type_id)));

    // Float casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float32_type_id), ndt::type(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float32_type_id), ndt::type(float64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_type_id), ndt::type(float32_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_type_id), ndt::type(float64_type_id)));

    // Complex Casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float32_type_id), ndt::type(complex_float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float32_type_id), ndt::type(complex_float64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_type_id), ndt::type(complex_float32_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_type_id), ndt::type(complex_float64_type_id)));

    // String casting
    // String conversions report false, so that assignments encodings
    // get validated on assignment
    EXPECT_FALSE(is_lossless_assignment(ndt::make_fixedstring(16, string_encoding_utf_16),
                            ndt::make_fixedstring(16, string_encoding_utf_16)));
    EXPECT_FALSE(is_lossless_assignment(ndt::make_fixedstring(16, string_encoding_utf_8),
                            ndt::make_fixedstring(12, string_encoding_utf_8)));
    EXPECT_FALSE(is_lossless_assignment(ndt::make_fixedstring(12, string_encoding_utf_32),
                            ndt::make_fixedstring(16, string_encoding_utf_32)));
    EXPECT_FALSE(is_lossless_assignment(ndt::make_fixedstring(16, string_encoding_utf_16),
                            ndt::make_fixedstring(16, string_encoding_utf_32)));
    EXPECT_FALSE(is_lossless_assignment(ndt::make_fixedstring(16, string_encoding_utf_8),
                            ndt::make_fixedstring(16, string_encoding_utf_16)));

    // Int -> UInt casting
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint64_type_id), ndt::type(int8_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint64_type_id), ndt::type(int64_type_id)));

    // UInt -> Int casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_type_id), ndt::type(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_type_id), ndt::type(uint16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int64_type_id), ndt::type(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int64_type_id), ndt::type(uint64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int32_type_id), ndt::type(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(int32_type_id), ndt::type(uint16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int32_type_id), ndt::type(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int32_type_id), ndt::type(int64_type_id)));

    // Int -> Float casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float32_type_id), ndt::type(int8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float32_type_id), ndt::type(int16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float32_type_id), ndt::type(int32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float32_type_id), ndt::type(int64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_type_id), ndt::type(int8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_type_id), ndt::type(int16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_type_id), ndt::type(int32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float64_type_id), ndt::type(int64_type_id)));

    // Int -> Complex casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float32_type_id), ndt::type(int8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float32_type_id), ndt::type(int16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float32_type_id), ndt::type(int32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float32_type_id), ndt::type(int64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_type_id), ndt::type(int8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_type_id), ndt::type(int16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_type_id), ndt::type(int32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float64_type_id), ndt::type(int64_type_id)));

    // UInt -> Float casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float32_type_id), ndt::type(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float32_type_id), ndt::type(uint16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float32_type_id), ndt::type(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float32_type_id), ndt::type(uint64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_type_id), ndt::type(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_type_id), ndt::type(uint16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(float64_type_id), ndt::type(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float64_type_id), ndt::type(uint64_type_id)));

    // UInt -> Complex casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float32_type_id), ndt::type(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float32_type_id), ndt::type(uint16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float32_type_id), ndt::type(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float32_type_id), ndt::type(uint64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_type_id), ndt::type(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_type_id), ndt::type(uint16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_type_id), ndt::type(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float64_type_id), ndt::type(uint64_type_id)));

    // Float -> Int casting
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int8_type_id),  ndt::type(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int16_type_id), ndt::type(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int32_type_id), ndt::type(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int64_type_id), ndt::type(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int8_type_id),  ndt::type(float64_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int16_type_id), ndt::type(float64_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int32_type_id), ndt::type(float64_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(int64_type_id), ndt::type(float64_type_id)));

    // Float -> UInt casting
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint8_type_id),  ndt::type(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint16_type_id), ndt::type(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint32_type_id), ndt::type(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint64_type_id), ndt::type(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint8_type_id),  ndt::type(float64_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint16_type_id), ndt::type(float64_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint32_type_id), ndt::type(float64_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(uint64_type_id), ndt::type(float64_type_id)));

    // Float -> Complex casting
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float32_type_id), ndt::type(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(complex_float32_type_id), ndt::type(float64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_type_id), ndt::type(float32_type_id)));
    EXPECT_TRUE( is_lossless_assignment(ndt::type(complex_float64_type_id), ndt::type(float64_type_id)));

    // Complex -> Float casting
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float32_type_id), ndt::type(complex_float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float32_type_id), ndt::type(complex_float64_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float64_type_id), ndt::type(complex_float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(ndt::type(float64_type_id), ndt::type(complex_float64_type_id)));
}

