#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <gtest/gtest.h>

#include "dnd/dtype_casting.hpp"

using namespace std;
using namespace dnd;

TEST(DTypeCasting, CanCastLosslessly) {
    // Boolean casting
    EXPECT_EQ(true, can_cast_losslessly(dtype(bool_type_id), dtype(bool_type_id)));

    // Signed int casting
    EXPECT_EQ(true,  can_cast_losslessly(dtype(int8_type_id),  dtype(int8_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(int8_type_id),  dtype(int16_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(int8_type_id),  dtype(int32_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(int8_type_id),  dtype(int64_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(int16_type_id), dtype(int8_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(int16_type_id), dtype(int16_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(int16_type_id), dtype(int32_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(int16_type_id), dtype(int64_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(int32_type_id), dtype(int8_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(int32_type_id), dtype(int16_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(int32_type_id), dtype(int32_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(int32_type_id), dtype(int64_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(int64_type_id), dtype(int8_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(int64_type_id), dtype(int16_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(int64_type_id), dtype(int32_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(int64_type_id), dtype(int64_type_id)));

    // Unsigned int casting
    EXPECT_EQ(true,  can_cast_losslessly(dtype(uint8_type_id),  dtype(uint8_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(uint8_type_id),  dtype(uint16_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(uint8_type_id),  dtype(uint32_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(uint8_type_id),  dtype(uint64_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(uint16_type_id), dtype(uint8_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(uint16_type_id), dtype(uint16_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(uint16_type_id), dtype(uint32_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(uint16_type_id), dtype(uint64_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(uint32_type_id), dtype(uint8_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(uint32_type_id), dtype(uint16_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(uint32_type_id), dtype(uint32_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(uint32_type_id), dtype(uint64_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(uint64_type_id), dtype(uint8_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(uint64_type_id), dtype(uint16_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(uint64_type_id), dtype(uint32_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(uint64_type_id), dtype(uint64_type_id)));

    // Float casting
    EXPECT_EQ(true,  can_cast_losslessly(dtype(float32_type_id), dtype(float32_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(float32_type_id), dtype(float64_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(float64_type_id), dtype(float32_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(float64_type_id), dtype(float64_type_id)));

    // String casting
    EXPECT_EQ(true, can_cast_losslessly(dtype(utf8_type_id, 16), dtype(utf8_type_id, 16)));
    EXPECT_EQ(true, can_cast_losslessly(dtype(utf8_type_id, 16), dtype(utf8_type_id, 12)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(utf8_type_id, 12), dtype(utf8_type_id, 16)));

    // Int -> UInt casting
    EXPECT_EQ(false, can_cast_losslessly(dtype(uint64_type_id), dtype(int8_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(uint64_type_id), dtype(int64_type_id)));

    // UInt -> Int casting
    EXPECT_EQ(true,  can_cast_losslessly(dtype(int64_type_id), dtype(uint8_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(int64_type_id), dtype(uint16_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(int64_type_id), dtype(uint32_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(int64_type_id), dtype(uint64_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(int32_type_id), dtype(uint8_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(int32_type_id), dtype(uint16_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(int32_type_id), dtype(uint32_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(int32_type_id), dtype(int64_type_id)));

    // Int -> Float casting
    EXPECT_EQ(true,  can_cast_losslessly(dtype(float32_type_id), dtype(int8_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(float32_type_id), dtype(int16_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(float32_type_id), dtype(int32_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(float32_type_id), dtype(int64_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(float64_type_id), dtype(int8_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(float64_type_id), dtype(int16_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(float64_type_id), dtype(int32_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(float64_type_id), dtype(int64_type_id)));

    // UInt -> Float casting
    EXPECT_EQ(true,  can_cast_losslessly(dtype(float32_type_id), dtype(uint8_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(float32_type_id), dtype(uint16_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(float32_type_id), dtype(uint32_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(float32_type_id), dtype(uint64_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(float64_type_id), dtype(uint8_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(float64_type_id), dtype(uint16_type_id)));
    EXPECT_EQ(true,  can_cast_losslessly(dtype(float64_type_id), dtype(uint32_type_id)));
    EXPECT_EQ(false, can_cast_losslessly(dtype(float64_type_id), dtype(uint64_type_id)));
}

