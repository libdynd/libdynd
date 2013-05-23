//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include "inc_gtest.hpp"

#include <dynd/dtype_assign.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(DTypeCasting, IsLosslessAssignment) {
    // Boolean casting
    EXPECT_TRUE(is_lossless_assignment(dtype(bool_type_id), dtype(bool_type_id)));

    // Signed int casting
    EXPECT_TRUE( is_lossless_assignment(dtype(int8_type_id),  dtype(bool_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int8_type_id),  dtype(int8_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int8_type_id),  dtype(int16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int8_type_id),  dtype(int32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int8_type_id),  dtype(int64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int16_type_id), dtype(bool_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int16_type_id), dtype(int8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int16_type_id), dtype(int16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int16_type_id), dtype(int32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int16_type_id), dtype(int64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int32_type_id), dtype(bool_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int32_type_id), dtype(int8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int32_type_id), dtype(int16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int32_type_id), dtype(int32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int32_type_id), dtype(int64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int64_type_id), dtype(bool_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int64_type_id), dtype(int8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int64_type_id), dtype(int16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int64_type_id), dtype(int32_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int64_type_id), dtype(int64_type_id)));

    // Unsigned int casting
    EXPECT_TRUE( is_lossless_assignment(dtype(uint8_type_id),  dtype(uint8_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(uint8_type_id),  dtype(uint16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(uint8_type_id),  dtype(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(uint8_type_id),  dtype(uint64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(uint16_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(uint16_type_id), dtype(uint16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(uint16_type_id), dtype(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(uint16_type_id), dtype(uint64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(uint32_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(uint32_type_id), dtype(uint16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(uint32_type_id), dtype(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(uint32_type_id), dtype(uint64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(uint64_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(uint64_type_id), dtype(uint16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(uint64_type_id), dtype(uint32_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(uint64_type_id), dtype(uint64_type_id)));

    // Float casting
    EXPECT_TRUE( is_lossless_assignment(dtype(float32_type_id), dtype(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(float32_type_id), dtype(float64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(float64_type_id), dtype(float32_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(float64_type_id), dtype(float64_type_id)));

    // Complex Casting
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float32_type_id), dtype(complex_float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(complex_float32_type_id), dtype(complex_float64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float64_type_id), dtype(complex_float32_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float64_type_id), dtype(complex_float64_type_id)));

    // String casting
    // String conversions report false, so that assignments encodings
    // get validated on assignment
    EXPECT_FALSE(is_lossless_assignment(make_fixedstring_dtype(16, string_encoding_utf_16),
                            make_fixedstring_dtype(16, string_encoding_utf_16)));
    EXPECT_FALSE(is_lossless_assignment(make_fixedstring_dtype(16, string_encoding_utf_8),
                            make_fixedstring_dtype(12, string_encoding_utf_8)));
    EXPECT_FALSE(is_lossless_assignment(make_fixedstring_dtype(12, string_encoding_utf_32),
                            make_fixedstring_dtype(16, string_encoding_utf_32)));
    EXPECT_FALSE(is_lossless_assignment(make_fixedstring_dtype(16, string_encoding_utf_16),
                            make_fixedstring_dtype(16, string_encoding_utf_32)));
    EXPECT_FALSE(is_lossless_assignment(make_fixedstring_dtype(16, string_encoding_utf_8),
                            make_fixedstring_dtype(16, string_encoding_utf_16)));

    // Int -> UInt casting
    EXPECT_FALSE(is_lossless_assignment(dtype(uint64_type_id), dtype(int8_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(uint64_type_id), dtype(int64_type_id)));

    // UInt -> Int casting
    EXPECT_TRUE( is_lossless_assignment(dtype(int64_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int64_type_id), dtype(uint16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int64_type_id), dtype(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int64_type_id), dtype(uint64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int32_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(int32_type_id), dtype(uint16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int32_type_id), dtype(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int32_type_id), dtype(int64_type_id)));

    // Int -> Float casting
    EXPECT_TRUE( is_lossless_assignment(dtype(float32_type_id), dtype(int8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(float32_type_id), dtype(int16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(float32_type_id), dtype(int32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(float32_type_id), dtype(int64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(float64_type_id), dtype(int8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(float64_type_id), dtype(int16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(float64_type_id), dtype(int32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(float64_type_id), dtype(int64_type_id)));

    // Int -> Complex casting
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float32_type_id), dtype(int8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float32_type_id), dtype(int16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(complex_float32_type_id), dtype(int32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(complex_float32_type_id), dtype(int64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float64_type_id), dtype(int8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float64_type_id), dtype(int16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float64_type_id), dtype(int32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(complex_float64_type_id), dtype(int64_type_id)));

    // UInt -> Float casting
    EXPECT_TRUE( is_lossless_assignment(dtype(float32_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(float32_type_id), dtype(uint16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(float32_type_id), dtype(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(float32_type_id), dtype(uint64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(float64_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(float64_type_id), dtype(uint16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(float64_type_id), dtype(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(float64_type_id), dtype(uint64_type_id)));

    // UInt -> Complex casting
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float32_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float32_type_id), dtype(uint16_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(complex_float32_type_id), dtype(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(complex_float32_type_id), dtype(uint64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float64_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float64_type_id), dtype(uint16_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float64_type_id), dtype(uint32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(complex_float64_type_id), dtype(uint64_type_id)));

    // Float -> Int casting
    EXPECT_FALSE(is_lossless_assignment(dtype(int8_type_id),  dtype(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int16_type_id), dtype(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int32_type_id), dtype(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int64_type_id), dtype(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int8_type_id),  dtype(float64_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int16_type_id), dtype(float64_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int32_type_id), dtype(float64_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(int64_type_id), dtype(float64_type_id)));

    // Float -> UInt casting
    EXPECT_FALSE(is_lossless_assignment(dtype(uint8_type_id),  dtype(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(uint16_type_id), dtype(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(uint32_type_id), dtype(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(uint64_type_id), dtype(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(uint8_type_id),  dtype(float64_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(uint16_type_id), dtype(float64_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(uint32_type_id), dtype(float64_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(uint64_type_id), dtype(float64_type_id)));

    // Float -> Complex casting
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float32_type_id), dtype(float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(complex_float32_type_id), dtype(float64_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float64_type_id), dtype(float32_type_id)));
    EXPECT_TRUE( is_lossless_assignment(dtype(complex_float64_type_id), dtype(float64_type_id)));

    // Complex -> Float casting
    EXPECT_FALSE(is_lossless_assignment(dtype(float32_type_id), dtype(complex_float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(float32_type_id), dtype(complex_float64_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(float64_type_id), dtype(complex_float32_type_id)));
    EXPECT_FALSE(is_lossless_assignment(dtype(float64_type_id), dtype(complex_float64_type_id)));
}

