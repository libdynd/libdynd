#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <gtest/gtest.h>

#include "dnd/dtype_assign.hpp"

using namespace std;
using namespace dnd;

TEST(DTypeAssign, FixedSizeTestsNoExcept) {
    dnd_bool v_b;
    int8_t v_i8;
    int16_t v_i16;
    int32_t v_i32;
    int64_t v_i64;
    uint8_t v_u8;
    uint16_t v_u16;
    uint32_t v_u32;
    uint64_t v_u64;
    float v_f32;
    double v_f64;

    dtype s_dt, d_dt;
    void *s_ptr;

    s_dt = dtype(bool_type_id);
    s_ptr = &v_b;
    v_b = true;
    dtype_assign_noexcept(&v_i8, s_ptr, dtype(int8_type_id), s_dt);
    EXPECT_EQ(1, v_i8);
    dtype_assign_noexcept(&v_i16, s_ptr, dtype(int16_type_id), s_dt);
    EXPECT_EQ(1, v_i16);
    dtype_assign_noexcept(&v_i32, s_ptr, dtype(int32_type_id), s_dt);
    EXPECT_EQ(1, v_i32);
    dtype_assign_noexcept(&v_i64, s_ptr, dtype(int64_type_id), s_dt);
    EXPECT_EQ(1, v_i64);
    dtype_assign_noexcept(&v_u8, s_ptr, dtype(uint8_type_id), s_dt);
    EXPECT_EQ(1, v_u8);
    dtype_assign_noexcept(&v_u16, s_ptr, dtype(uint16_type_id), s_dt);
    EXPECT_EQ(1, v_u16);
    dtype_assign_noexcept(&v_u32, s_ptr, dtype(uint32_type_id), s_dt);
    EXPECT_EQ(1, v_u32);
    dtype_assign_noexcept(&v_u64, s_ptr, dtype(uint64_type_id), s_dt);
    EXPECT_EQ(1, v_u64);
    dtype_assign_noexcept(&v_f32, s_ptr, dtype(float32_type_id), s_dt);
    EXPECT_EQ(1, v_f32);
    dtype_assign_noexcept(&v_f64, s_ptr, dtype(float64_type_id), s_dt);
    EXPECT_EQ(1, v_f64);

    s_dt = dtype(int8_type_id);
    s_ptr = &v_i8;
    v_i8 = 127;
    dtype_assign_noexcept(&v_b, s_ptr, dtype(bool_type_id), s_dt);
    EXPECT_EQ(true, v_b);
    dtype_assign_noexcept(&v_i16, s_ptr, dtype(int16_type_id), s_dt);
    EXPECT_EQ(127, v_i16);
    dtype_assign_noexcept(&v_i32, s_ptr, dtype(int32_type_id), s_dt);
    EXPECT_EQ(127, v_i32);
    dtype_assign_noexcept(&v_i64, s_ptr, dtype(int64_type_id), s_dt);
    EXPECT_EQ(127, v_i64);
    dtype_assign_noexcept(&v_u8, s_ptr, dtype(uint8_type_id), s_dt);
    EXPECT_EQ(127, v_u8);
    dtype_assign_noexcept(&v_u16, s_ptr, dtype(uint16_type_id), s_dt);
    EXPECT_EQ(127, v_u16);
    dtype_assign_noexcept(&v_u32, s_ptr, dtype(uint32_type_id), s_dt);
    EXPECT_EQ(127, v_u32);
    dtype_assign_noexcept(&v_u64, s_ptr, dtype(uint64_type_id), s_dt);
    EXPECT_EQ(127, v_u64);
    dtype_assign_noexcept(&v_f32, s_ptr, dtype(float32_type_id), s_dt);
    EXPECT_EQ(127, v_f32);
    dtype_assign_noexcept(&v_f64, s_ptr, dtype(float64_type_id), s_dt);
    EXPECT_EQ(127, v_f64);

    s_dt = dtype(float64_type_id);
    s_ptr = &v_f64;
    v_f64 = -10.25;
    dtype_assign_noexcept(&v_b, s_ptr, dtype(bool_type_id), s_dt);
    EXPECT_EQ(v_b, true);
    dtype_assign_noexcept(&v_i16, s_ptr, dtype(int16_type_id), s_dt);
    EXPECT_EQ(v_i16, -10);
    dtype_assign_noexcept(&v_i32, s_ptr, dtype(int32_type_id), s_dt);
    EXPECT_EQ(v_i32, -10);
    dtype_assign_noexcept(&v_i64, s_ptr, dtype(int64_type_id), s_dt);
    EXPECT_EQ(v_i64, -10);
    dtype_assign_noexcept(&v_u8, s_ptr, dtype(uint8_type_id), s_dt);
    EXPECT_EQ(v_u8, (uint8_t)-10);
    dtype_assign_noexcept(&v_u16, s_ptr, dtype(uint16_type_id), s_dt);
    EXPECT_EQ(v_u16, (uint16_t)-10);
    dtype_assign_noexcept(&v_u32, s_ptr, dtype(uint32_type_id), s_dt);
    EXPECT_EQ(v_u32, (uint32_t)-10);
    dtype_assign_noexcept(&v_u64, s_ptr, dtype(uint64_type_id), s_dt);
    EXPECT_EQ(v_u64, (uint64_t)-10);
    dtype_assign_noexcept(&v_f32, s_ptr, dtype(float32_type_id), s_dt);
    EXPECT_EQ(v_f32, -10.25);
    dtype_assign_noexcept(&v_f64, s_ptr, dtype(float64_type_id), s_dt);
    EXPECT_EQ(v_f64, -10.25);
}

TEST(DTypeAssign, FixedSizeTests) {
    dnd_bool v_b;
    int8_t v_i8;
    int16_t v_i16;
    int32_t v_i32;
    int64_t v_i64;
    uint8_t v_u8;
    uint16_t v_u16;
    uint32_t v_u32;
    uint64_t v_u64;
    float v_f32;
    double v_f64;

    dtype s_dt, d_dt;
    void *s_ptr;

    s_dt = dtype(bool_type_id);
    s_ptr = &v_b;
    v_b = true;
    dtype_assign(&v_i8, s_ptr, dtype(int8_type_id), s_dt);
    EXPECT_EQ(1, v_i8);
    dtype_assign(&v_i16, s_ptr, dtype(int16_type_id), s_dt);
    EXPECT_EQ(1, v_i16);
    dtype_assign(&v_i32, s_ptr, dtype(int32_type_id), s_dt);
    EXPECT_EQ(1, v_i32);
    dtype_assign(&v_i64, s_ptr, dtype(int64_type_id), s_dt);
    EXPECT_EQ(1, v_i64);
    dtype_assign(&v_u8, s_ptr, dtype(uint8_type_id), s_dt);
    EXPECT_EQ(1, v_u8);
    dtype_assign(&v_u16, s_ptr, dtype(uint16_type_id), s_dt);
    EXPECT_EQ(1, v_u16);
    dtype_assign(&v_u32, s_ptr, dtype(uint32_type_id), s_dt);
    EXPECT_EQ(1, v_u32);
    dtype_assign(&v_u64, s_ptr, dtype(uint64_type_id), s_dt);
    EXPECT_EQ(1, v_u64);
    dtype_assign(&v_f32, s_ptr, dtype(float32_type_id), s_dt);
    EXPECT_EQ(1, v_f32);
    dtype_assign(&v_f64, s_ptr, dtype(float64_type_id), s_dt);
    EXPECT_EQ(1, v_f64);

    s_dt = dtype(int8_type_id);
    s_ptr = &v_i8;
    v_i8 = 127;
    EXPECT_THROW(dtype_assign(&v_b, s_ptr, dtype(bool_type_id), s_dt), runtime_error);
    dtype_assign(&v_i16, s_ptr, dtype(int16_type_id), s_dt);
    EXPECT_EQ(127, v_i16);
    dtype_assign(&v_i32, s_ptr, dtype(int32_type_id), s_dt);
    EXPECT_EQ(127, v_i32);
    dtype_assign(&v_i64, s_ptr, dtype(int64_type_id), s_dt);
    EXPECT_EQ(127, v_i64);
    dtype_assign(&v_u8, s_ptr, dtype(uint8_type_id), s_dt);
    EXPECT_EQ(127, v_u8);
    dtype_assign(&v_u16, s_ptr, dtype(uint16_type_id), s_dt);
    EXPECT_EQ(127, v_u16);
    dtype_assign(&v_u32, s_ptr, dtype(uint32_type_id), s_dt);
    EXPECT_EQ(127, v_u32);
    dtype_assign(&v_u64, s_ptr, dtype(uint64_type_id), s_dt);
    EXPECT_EQ(127, v_u64);
    dtype_assign(&v_f32, s_ptr, dtype(float32_type_id), s_dt);
    EXPECT_EQ(127, v_f32);
    dtype_assign(&v_f64, s_ptr, dtype(float64_type_id), s_dt);
    EXPECT_EQ(127, v_f64);

    v_i8 = -33;
    EXPECT_THROW(dtype_assign(&v_b, s_ptr, dtype(bool_type_id), s_dt), runtime_error);
    v_i8 = -1;
    EXPECT_THROW(dtype_assign(&v_b, s_ptr, dtype(bool_type_id), s_dt), runtime_error);
    v_i8 = 2;
    EXPECT_THROW(dtype_assign(&v_b, s_ptr, dtype(bool_type_id), s_dt), runtime_error);
    v_i8 = 0;
    dtype_assign(&v_b, s_ptr, dtype(bool_type_id), s_dt);
    EXPECT_EQ(false, v_b);
    v_i8 = 1;
    dtype_assign(&v_b, s_ptr, dtype(bool_type_id), s_dt);
    EXPECT_EQ(true, v_b);

    s_dt = dtype(float64_type_id);
    s_ptr = &v_f64;
    v_f64 = -10.25;
    EXPECT_THROW(dtype_assign(&v_b, s_ptr, dtype(bool_type_id), s_dt), runtime_error);
    EXPECT_THROW(dtype_assign(&v_i8, s_ptr, dtype(int8_type_id), s_dt), runtime_error);
    EXPECT_THROW(dtype_assign(&v_i16, s_ptr, dtype(int16_type_id), s_dt), runtime_error);
    EXPECT_THROW(dtype_assign(&v_i32, s_ptr, dtype(int32_type_id), s_dt), runtime_error);
    EXPECT_THROW(dtype_assign(&v_i64, s_ptr, dtype(int64_type_id), s_dt), runtime_error);
    EXPECT_THROW(dtype_assign(&v_u8, s_ptr, dtype(uint8_type_id), s_dt), runtime_error);
    EXPECT_THROW(dtype_assign(&v_u16, s_ptr, dtype(uint16_type_id), s_dt), runtime_error);
    EXPECT_THROW(dtype_assign(&v_u32, s_ptr, dtype(uint32_type_id), s_dt), runtime_error);
    EXPECT_THROW(dtype_assign(&v_u64, s_ptr, dtype(uint64_type_id), s_dt), runtime_error);
    dtype_assign(&v_f32, s_ptr, dtype(float32_type_id), s_dt);
    EXPECT_EQ(-10.25, v_f32);
    dtype_assign(&v_f64, s_ptr, dtype(float64_type_id), s_dt);
    EXPECT_EQ(-10.25, v_f64);

    // dtype_assign checks that the float64 -> float32value gets converted exactly
    v_f64 = 1 / 3.0;
    EXPECT_THROW(dtype_assign(&v_f32, s_ptr, dtype(float32_type_id), s_dt), runtime_error);
    v_f64 = 1 / 3.0f;
    dtype_assign(&v_f32, s_ptr, dtype(float32_type_id), s_dt);
    EXPECT_EQ(v_f64, v_f32);
}

