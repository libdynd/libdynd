#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include "inc_gtest.hpp"

#include "dnd/dtype_assign.hpp"

using namespace std;
using namespace dnd;

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
    EXPECT_TRUE(is_lossless_assignment(dtype(utf8_type_id, 16), dtype(utf8_type_id, 16)));
    EXPECT_TRUE(is_lossless_assignment(dtype(utf8_type_id, 16), dtype(utf8_type_id, 12)));
    EXPECT_FALSE(is_lossless_assignment(dtype(utf8_type_id, 12), dtype(utf8_type_id, 16)));

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

