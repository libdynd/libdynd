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

TEST(DTypeAssign, FixedSizeTestsStridedNoExcept) {
    dnd_bool v_b[4];
    int8_t v_i8[4];
    int16_t v_i16[4];
    int32_t v_i32[4];
    int64_t v_i64[4];
    uint8_t v_u8[4];
    uint16_t v_u16[4];
    uint32_t v_u32[4];
    uint64_t v_u64[4];
    float v_f32[4];
    double v_f64[4];

    dtype s_dt, d_dt;
    void *s_ptr;
    intptr_t s_stride;

    s_dt = dtype(bool_type_id);
    s_ptr = v_b;
    s_stride = sizeof(v_b[0]);
    v_b[0] = true; v_b[1] = true; v_b[2] = false; v_b[3] = true;
    dtype_strided_assign_noexcept(v_i8, sizeof(v_i8[0]), s_ptr, s_stride, 4, dtype(int8_type_id), s_dt);
    EXPECT_EQ(1, v_i8[0]); EXPECT_EQ(1, v_i8[1]);
    EXPECT_EQ(0, v_i8[2]); EXPECT_EQ(1, v_i8[3]);
    dtype_strided_assign_noexcept(v_i16, sizeof(v_i16[0]), s_ptr, s_stride, 4, dtype(int16_type_id), s_dt);
    EXPECT_EQ(1, v_i16[0]); EXPECT_EQ(1, v_i16[1]);
    EXPECT_EQ(0, v_i16[2]); EXPECT_EQ(1, v_i16[3]);
    dtype_strided_assign_noexcept(v_i32, sizeof(v_i32[0]), s_ptr, s_stride, 4, dtype(int32_type_id), s_dt);
    EXPECT_EQ(1, v_i32[0]); EXPECT_EQ(1, v_i32[1]);
    EXPECT_EQ(0, v_i32[2]); EXPECT_EQ(1, v_i32[3]);
    dtype_strided_assign_noexcept(v_i64, sizeof(v_i64[0]), s_ptr, s_stride, 4, dtype(int64_type_id), s_dt);
    EXPECT_EQ(1, v_i64[0]); EXPECT_EQ(1, v_i64[1]);
    EXPECT_EQ(0, v_i64[2]); EXPECT_EQ(1, v_i64[3]);
    dtype_strided_assign_noexcept(v_u8, sizeof(v_u8[0]), s_ptr, s_stride, 4, dtype(uint8_type_id), s_dt);
    EXPECT_EQ(1, v_u8[0]); EXPECT_EQ(1, v_u8[1]);
    EXPECT_EQ(0, v_u8[2]); EXPECT_EQ(1, v_u8[3]);
    dtype_strided_assign_noexcept(v_u16, sizeof(v_u16[0]), s_ptr, s_stride, 4, dtype(uint16_type_id), s_dt);
    EXPECT_EQ(1, v_u16[0]); EXPECT_EQ(1, v_u16[1]);
    EXPECT_EQ(0, v_u16[2]); EXPECT_EQ(1, v_u16[3]);
    dtype_strided_assign_noexcept(v_u32, sizeof(v_u32[0]), s_ptr, s_stride, 4, dtype(uint32_type_id), s_dt);
    EXPECT_EQ(1, v_u32[0]); EXPECT_EQ(1, v_u32[1]);
    EXPECT_EQ(0, v_u32[2]); EXPECT_EQ(1, v_u32[3]);
    dtype_strided_assign_noexcept(v_u64, sizeof(v_u64[0]), s_ptr, s_stride, 4, dtype(uint64_type_id), s_dt);
    EXPECT_EQ(1, v_u64[0]); EXPECT_EQ(1, v_u64[1]);
    EXPECT_EQ(0, v_u64[2]); EXPECT_EQ(1, v_u64[3]);
    dtype_strided_assign_noexcept(v_f32, sizeof(v_f32[0]), s_ptr, s_stride, 4, dtype(float32_type_id), s_dt);
    EXPECT_EQ(1, v_f32[0]); EXPECT_EQ(1, v_f32[1]);
    EXPECT_EQ(0, v_f32[2]); EXPECT_EQ(1, v_f32[3]);
    dtype_strided_assign_noexcept(v_f64, sizeof(v_f64[0]), s_ptr, s_stride, 4, dtype(float64_type_id), s_dt);
    EXPECT_EQ(1, v_f64[0]); EXPECT_EQ(1, v_f64[1]);
    EXPECT_EQ(0, v_f64[2]); EXPECT_EQ(1, v_f64[3]);

    s_dt = dtype(int8_type_id);
    s_ptr = v_i8;
    s_stride = sizeof(v_i8[0]);
    v_i8[0] = 127; v_i8[1] = 0; v_i8[2] = -128; v_i8[3] = -10;
    dtype_strided_assign_noexcept(v_b, sizeof(v_b[0]), s_ptr, s_stride, 4, dtype(bool_type_id), s_dt);
    EXPECT_EQ(true, v_b[0]); EXPECT_EQ(false, v_b[1]);
    EXPECT_EQ(true, v_b[2]); EXPECT_EQ(true, v_b[3]);
    dtype_strided_assign_noexcept(v_i16, sizeof(v_i16[0]), s_ptr, s_stride, 4, dtype(int16_type_id), s_dt);
    EXPECT_EQ(127, v_i16[0]); EXPECT_EQ(0, v_i16[1]);
    EXPECT_EQ(-128, v_i16[2]); EXPECT_EQ(-10, v_i16[3]);
    dtype_strided_assign_noexcept(v_i32, sizeof(v_i32[0]), s_ptr, s_stride, 4, dtype(int32_type_id), s_dt);
    EXPECT_EQ(127, v_i32[0]); EXPECT_EQ(0, v_i32[1]);
    EXPECT_EQ(-128, v_i32[2]); EXPECT_EQ(-10, v_i32[3]);
    dtype_strided_assign_noexcept(v_i64, sizeof(v_i64[0]), s_ptr, s_stride, 4, dtype(int64_type_id), s_dt);
    EXPECT_EQ(127, v_i64[0]); EXPECT_EQ(0, v_i64[1]);
    EXPECT_EQ(-128, v_i64[2]); EXPECT_EQ(-10, v_i64[3]);
    dtype_strided_assign_noexcept(v_u8, sizeof(v_u8[0]), s_ptr, s_stride, 4, dtype(uint8_type_id), s_dt);
    EXPECT_EQ(127, v_u8[0]); EXPECT_EQ(0, v_u8[1]);
    EXPECT_EQ((uint8_t)-128, v_u8[2]); EXPECT_EQ((uint8_t)-10, v_u8[3]);
    dtype_strided_assign_noexcept(v_u16, sizeof(v_u16[0]), s_ptr, s_stride, 4, dtype(uint16_type_id), s_dt);
    EXPECT_EQ(127, v_u16[0]); EXPECT_EQ(0, v_u16[1]);
    EXPECT_EQ((uint16_t)-128, v_u16[2]); EXPECT_EQ((uint16_t)-10, v_u16[3]);
    dtype_strided_assign_noexcept(v_u32, sizeof(v_u32[0]), s_ptr, s_stride, 4, dtype(uint32_type_id), s_dt);
    EXPECT_EQ(127, v_u32[0]); EXPECT_EQ(0, v_u32[1]);
    EXPECT_EQ((uint32_t)-128, v_u32[2]); EXPECT_EQ((uint32_t)-10, v_u32[3]);
    dtype_strided_assign_noexcept(v_u64, sizeof(v_u64[0]), s_ptr, s_stride, 4, dtype(uint64_type_id), s_dt);
    EXPECT_EQ(127, v_u64[0]); EXPECT_EQ(0, v_u64[1]);
    EXPECT_EQ((uint64_t)-128, v_u64[2]); EXPECT_EQ((uint64_t)-10, v_u64[3]);
    dtype_strided_assign_noexcept(v_f32, sizeof(v_f32[0]), s_ptr, s_stride, 4, dtype(float32_type_id), s_dt);
    EXPECT_EQ(127, v_f32[0]); EXPECT_EQ(0, v_f32[1]);
    EXPECT_EQ(-128, v_f32[2]); EXPECT_EQ(-10, v_f32[3]);
    dtype_strided_assign_noexcept(v_f64, sizeof(v_f64[0]), s_ptr, s_stride, 4, dtype(float64_type_id), s_dt);
    EXPECT_EQ(127, v_f64[0]); EXPECT_EQ(0, v_f64[1]);
    EXPECT_EQ(-128, v_f64[2]); EXPECT_EQ(-10, v_f64[3]);

    s_dt = dtype(float64_type_id);
    s_ptr = v_f64;
    s_stride = 2*sizeof(v_f64[0]);
    v_f64[0] = -10.25; v_f64[1] = 2.25;
    v_f64[2] = 0.0; v_f64[3] = -5.5;
    dtype_strided_assign_noexcept(v_b, sizeof(v_b[0]), s_ptr, s_stride, 2, dtype(bool_type_id), s_dt);
    EXPECT_EQ(true, v_b[0]); EXPECT_EQ(false, v_b[1]);
    dtype_strided_assign_noexcept(v_i8, sizeof(v_i8[0]), s_ptr, s_stride, 2, dtype(int8_type_id), s_dt);
    EXPECT_EQ(-10, v_i8[0]); EXPECT_EQ(0, v_i8[1]);
    dtype_strided_assign_noexcept(v_i16, sizeof(v_i16[0]), s_ptr, s_stride, 2, dtype(int16_type_id), s_dt);
    EXPECT_EQ(-10, v_i16[0]); EXPECT_EQ(0, v_i16[1]);
    dtype_strided_assign_noexcept(v_i32, sizeof(v_i32[0]), s_ptr, s_stride, 2, dtype(int32_type_id), s_dt);
    EXPECT_EQ(-10, v_i32[0]); EXPECT_EQ(0, v_i32[1]);
    dtype_strided_assign_noexcept(v_i64, sizeof(v_i64[0]), s_ptr, s_stride, 2, dtype(int64_type_id), s_dt);
    EXPECT_EQ(-10, v_i64[0]); EXPECT_EQ(0, v_i64[1]);
    dtype_strided_assign_noexcept(v_u8, sizeof(v_u8[0]), s_ptr, s_stride, 2, dtype(uint8_type_id), s_dt);
    EXPECT_EQ((uint8_t)-10, v_u8[0]); EXPECT_EQ(0, v_u8[1]);
    dtype_strided_assign_noexcept(v_u16, sizeof(v_u16[0]), s_ptr, s_stride, 2, dtype(uint16_type_id), s_dt);
    EXPECT_EQ((uint16_t)-10, v_u16[0]); EXPECT_EQ(0, v_u16[1]);
    dtype_strided_assign_noexcept(v_u32, sizeof(v_u32[0]), s_ptr, s_stride, 2, dtype(uint32_type_id), s_dt);
    EXPECT_EQ((uint32_t)-10, v_u32[0]); EXPECT_EQ(0, v_u32[1]);
    dtype_strided_assign_noexcept(v_u64, sizeof(v_u64[0]), s_ptr, s_stride, 2, dtype(uint64_type_id), s_dt);
    EXPECT_EQ((uint64_t)-10, v_u64[0]); EXPECT_EQ(0, v_u64[1]);
    dtype_strided_assign_noexcept(v_f32, sizeof(v_f32[0]), s_ptr, s_stride, 2, dtype(float32_type_id), s_dt);
    EXPECT_EQ(-10.25, v_f32[0]); EXPECT_EQ(0, v_f32[1]);
    dtype_strided_assign_noexcept(v_f64, sizeof(v_f64[0]), s_ptr, s_stride, 2, dtype(float64_type_id), s_dt);
    EXPECT_EQ(-10.25, v_f64[0]); EXPECT_EQ(0, v_f64[1]);
}


