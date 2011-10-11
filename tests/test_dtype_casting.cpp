#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include "inc_gtest.hpp"

#include "dnd/dtype_casting.hpp"

using namespace std;
using namespace dnd;

TEST(DTypeCasting, CanCastLossless) {
    // Boolean casting
    EXPECT_TRUE(can_cast_lossless(dtype(bool_type_id), dtype(bool_type_id)));

    // Signed int casting
    EXPECT_TRUE( can_cast_lossless(dtype(int8_type_id),  dtype(int8_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(int8_type_id),  dtype(int16_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(int8_type_id),  dtype(int32_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(int8_type_id),  dtype(int64_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(int16_type_id), dtype(int8_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(int16_type_id), dtype(int16_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(int16_type_id), dtype(int32_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(int16_type_id), dtype(int64_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(int32_type_id), dtype(int8_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(int32_type_id), dtype(int16_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(int32_type_id), dtype(int32_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(int32_type_id), dtype(int64_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(int64_type_id), dtype(int8_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(int64_type_id), dtype(int16_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(int64_type_id), dtype(int32_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(int64_type_id), dtype(int64_type_id)));

    // Unsigned int casting
    EXPECT_TRUE( can_cast_lossless(dtype(uint8_type_id),  dtype(uint8_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(uint8_type_id),  dtype(uint16_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(uint8_type_id),  dtype(uint32_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(uint8_type_id),  dtype(uint64_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(uint16_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(uint16_type_id), dtype(uint16_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(uint16_type_id), dtype(uint32_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(uint16_type_id), dtype(uint64_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(uint32_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(uint32_type_id), dtype(uint16_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(uint32_type_id), dtype(uint32_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(uint32_type_id), dtype(uint64_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(uint64_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(uint64_type_id), dtype(uint16_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(uint64_type_id), dtype(uint32_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(uint64_type_id), dtype(uint64_type_id)));

    // Float casting
    EXPECT_TRUE( can_cast_lossless(dtype(float32_type_id), dtype(float32_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(float32_type_id), dtype(float64_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(float64_type_id), dtype(float32_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(float64_type_id), dtype(float64_type_id)));

    // String casting
    EXPECT_TRUE(can_cast_lossless(dtype(utf8_type_id, 16), dtype(utf8_type_id, 16)));
    EXPECT_TRUE(can_cast_lossless(dtype(utf8_type_id, 16), dtype(utf8_type_id, 12)));
    EXPECT_FALSE(can_cast_lossless(dtype(utf8_type_id, 12), dtype(utf8_type_id, 16)));

    // Int -> UInt casting
    EXPECT_FALSE(can_cast_lossless(dtype(uint64_type_id), dtype(int8_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(uint64_type_id), dtype(int64_type_id)));

    // UInt -> Int casting
    EXPECT_TRUE( can_cast_lossless(dtype(int64_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(int64_type_id), dtype(uint16_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(int64_type_id), dtype(uint32_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(int64_type_id), dtype(uint64_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(int32_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(int32_type_id), dtype(uint16_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(int32_type_id), dtype(uint32_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(int32_type_id), dtype(int64_type_id)));

    // Int -> Float casting
    EXPECT_TRUE( can_cast_lossless(dtype(float32_type_id), dtype(int8_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(float32_type_id), dtype(int16_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(float32_type_id), dtype(int32_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(float32_type_id), dtype(int64_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(float64_type_id), dtype(int8_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(float64_type_id), dtype(int16_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(float64_type_id), dtype(int32_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(float64_type_id), dtype(int64_type_id)));

    // UInt -> Float casting
    EXPECT_TRUE( can_cast_lossless(dtype(float32_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(float32_type_id), dtype(uint16_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(float32_type_id), dtype(uint32_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(float32_type_id), dtype(uint64_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(float64_type_id), dtype(uint8_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(float64_type_id), dtype(uint16_type_id)));
    EXPECT_TRUE( can_cast_lossless(dtype(float64_type_id), dtype(uint32_type_id)));
    EXPECT_FALSE(can_cast_lossless(dtype(float64_type_id), dtype(uint64_type_id)));
}

