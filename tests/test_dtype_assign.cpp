#include <iostream>
#include <stdexcept>
#include <cmath>
#include <stdint.h>
#include "inc_gtest.hpp"

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
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_none); \
            EXPECT_EQ(m, v)
    ONE_TEST(int8_type_id, v_i8, 1);
    ONE_TEST(int16_type_id, v_i16, 1);
    ONE_TEST(int32_type_id, v_i32, 1);
    ONE_TEST(int64_type_id, v_i64, 1);
    ONE_TEST(uint8_type_id, v_u8, 1u);
    ONE_TEST(uint16_type_id, v_u16, 1u);
    ONE_TEST(uint32_type_id, v_u32, 1u);
    ONE_TEST(uint64_type_id, v_u64, 1u);
    ONE_TEST(float32_type_id, v_f32, 1);
    ONE_TEST(float64_type_id, v_f64, 1);
#undef ONE_TEST

    s_dt = dtype(int8_type_id);
    s_ptr = &v_i8;
    v_i8 = 127;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_none); \
            EXPECT_EQ(m, v)
    ONE_TEST(bool_type_id, v_b, true);
    ONE_TEST(int16_type_id, v_i16, 127);
    ONE_TEST(int32_type_id, v_i32, 127);
    ONE_TEST(int64_type_id, v_i64, 127);
    ONE_TEST(uint8_type_id, v_u8, 127u);
    ONE_TEST(uint16_type_id, v_u16, 127u);
    ONE_TEST(uint32_type_id, v_u32, 127u);
    ONE_TEST(uint64_type_id, v_u64, 127u);
    ONE_TEST(float32_type_id, v_f32, 127u);
    ONE_TEST(float64_type_id, v_f64, 127u);
#undef ONE_TEST

    s_dt = dtype(float64_type_id);
    s_ptr = &v_f64;
    v_f64 = -10.25;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_none); \
            EXPECT_EQ(m, v)
    ONE_TEST(bool_type_id, v_b, true);
    ONE_TEST(int8_type_id, v_i8, -10);
    ONE_TEST(int16_type_id, v_i16, -10);
    ONE_TEST(int32_type_id, v_i32, -10);
    ONE_TEST(int64_type_id, v_i64, -10);
    ONE_TEST(uint8_type_id, v_u8, (uint8_t)-10);
    ONE_TEST(uint16_type_id, v_u16, (uint16_t)-10);
    ONE_TEST(uint32_type_id, v_u32, (uint32_t)-10);
    ONE_TEST(uint64_type_id, v_u64, (uint64_t)-10);
    ONE_TEST(float32_type_id, v_f32, -10.25);
#undef ONE_TEST
}

TEST(DTypeAssign, FixedSizeTests_Bool) {
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
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), &v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
    ONE_TEST(int8_type_id, v_i8, 1);
    ONE_TEST(int16_type_id, v_i16, 1);
    ONE_TEST(int32_type_id, v_i32, 1);
    ONE_TEST(int64_type_id, v_i64, 1);
    ONE_TEST(uint8_type_id, v_u8, 1u);
    ONE_TEST(uint16_type_id, v_u16, 1u);
    ONE_TEST(uint32_type_id, v_u32, 1u);
    ONE_TEST(uint64_type_id, v_u64, 1u);
    ONE_TEST(float32_type_id, v_f32, 1);
    ONE_TEST(float64_type_id, v_f64, 1);
#undef ONE_TEST
}

TEST(DTypeAssign, FixedSizeTests_Int8) {
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

    s_dt = dtype(int8_type_id);
    s_ptr = &v_i8;
    v_i8 = 127;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), &v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
    EXPECT_THROW(dtype_assign(dtype(bool_type_id), &v_b, s_dt, s_ptr), runtime_error);
    ONE_TEST(int16_type_id, v_i16, 127);
    ONE_TEST(int32_type_id, v_i32, 127);
    ONE_TEST(int64_type_id, v_i64, 127);
    ONE_TEST(uint8_type_id, v_u8, 127u);
    ONE_TEST(uint16_type_id, v_u16, 127u);
    ONE_TEST(uint32_type_id, v_u32, 127u);
    ONE_TEST(uint64_type_id, v_u64, 127u);
    ONE_TEST(float32_type_id, v_f32, 127);
    ONE_TEST(float64_type_id, v_f64, 127);
#undef ONE_TEST

    v_i8 = -33;
    EXPECT_THROW(dtype_assign(dtype(bool_type_id), &v_b, s_dt, s_ptr), runtime_error);
    v_i8 = -1;
    EXPECT_THROW(dtype_assign(dtype(bool_type_id), &v_b, s_dt, s_ptr), runtime_error);
    v_i8 = 2;
    EXPECT_THROW(dtype_assign(dtype(bool_type_id), &v_b, s_dt, s_ptr), runtime_error);
    v_i8 = 0;
    dtype_assign(dtype(bool_type_id), &v_b, s_dt, s_ptr);
    EXPECT_FALSE(v_b);
    v_i8 = 1;
    dtype_assign(dtype(bool_type_id), &v_b, s_dt, s_ptr);
    EXPECT_TRUE(v_b);
}

TEST(DTypeAssign, FixedSizeTests_Float64) {
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
    const char *s_ptr;

    s_dt = dtype(float64_type_id);
    s_ptr = reinterpret_cast<char *>(&v_f64);
    v_f64 = -10.25;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), &v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr), runtime_error)
    ONE_TEST_THROW(bool_type_id, v_b);
    ONE_TEST_THROW(int8_type_id, v_i8);
    ONE_TEST_THROW(int16_type_id, v_i16);
    ONE_TEST_THROW(int32_type_id, v_i32);
    ONE_TEST_THROW(int64_type_id, v_i64);
    ONE_TEST_THROW(uint8_type_id, v_u8);
    ONE_TEST_THROW(uint16_type_id, v_u16);
    ONE_TEST_THROW(uint32_type_id, v_u32);
    ONE_TEST_THROW(uint64_type_id, v_u64);
    ONE_TEST(float32_type_id, v_f32, -10.25);
#undef ONE_TEST
#undef ONE_TEST_THROW

    // dtype_assign checks that the float64 -> float32value gets converted exactly
    // when using the assign_error_inexact mode
    v_f64 = 1 / 3.0;
    EXPECT_THROW(dtype_assign(dtype(float32_type_id), &v_f32, s_dt, s_ptr, assign_error_inexact),
                                                                                runtime_error);
    dtype_assign(dtype(float32_type_id), &v_f32, s_dt, s_ptr, assign_error_fractional);
    EXPECT_EQ((float)v_f64, v_f32);

    // Since this is a float -> double conversion, it should be exact coming back to float
    v_f64 = 1 / 3.0f;
    dtype_assign(dtype(float32_type_id), &v_f32, s_dt, s_ptr, assign_error_inexact);
    EXPECT_EQ(v_f64, v_f32);

    // This should overflow converting to float
    v_f64 = -1.5e250;
    EXPECT_THROW(dtype_assign(dtype(float32_type_id), &v_f32, s_dt, s_ptr, assign_error_inexact),
                                                                                runtime_error);
    EXPECT_THROW(dtype_assign(dtype(float32_type_id), &v_f32, s_dt, s_ptr, assign_error_fractional),
                                                                                runtime_error);
    EXPECT_THROW(dtype_assign(dtype(float32_type_id), &v_f32, s_dt, s_ptr, assign_error_overflow),
                                                                                runtime_error);
    dtype_assign(dtype(float32_type_id), &v_f32, s_dt, s_ptr, assign_error_none);
#ifdef _WIN32
    EXPECT_TRUE(_fpclass(v_f32) == _FPCLASS_NINF);
#else
    EXPECT_TRUE(isinf(v_f32));
#endif
    EXPECT_TRUE(v_f32 < 0);
}

TEST(DTypeAssign, FixedSizeTestsStridedNoExcept_Bool) {
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
#define ONE_TEST(type, v) \
            dtype_strided_assign(make_dtype<type>(), v, sizeof(v[0]), s_dt, s_ptr, s_stride, 4, assign_error_none); \
            EXPECT_EQ((type)1, v[0]); EXPECT_EQ((type)1, v[1]); \
            EXPECT_EQ((type)0, v[2]); EXPECT_EQ((type)1, v[3])
    ONE_TEST(int8_t, v_i8);
    ONE_TEST(int16_t, v_i16);
    ONE_TEST(int32_t, v_i32);
    ONE_TEST(int64_t, v_i64);
    ONE_TEST(uint8_t, v_u8);
    ONE_TEST(uint16_t, v_u16);
    ONE_TEST(uint32_t, v_u32);
    ONE_TEST(uint64_t, v_u64);
    ONE_TEST(float, v_f32);
    ONE_TEST(double, v_f64);
#undef ONE_TEST
}

TEST(DTypeAssign, FixedSizeTestsStridedNoExcept_Int8) {
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

    s_dt = dtype(int8_type_id);
    s_ptr = v_i8;
    s_stride = sizeof(v_i8[0]);
    v_i8[0] = 127; v_i8[1] = 0; v_i8[2] = -128; v_i8[3] = -10;
#define ONE_TEST(tid, v, m0, m1, m2, m3) \
            dtype_strided_assign(dtype(tid), v, sizeof(v[0]), s_dt, s_ptr, s_stride, 4, assign_error_none); \
            EXPECT_EQ(m0, v[0]); EXPECT_EQ(m1, v[1]); \
            EXPECT_EQ(m2, v[2]); EXPECT_EQ(m3, v[3])

    dtype_strided_assign(dtype(bool_type_id), v_b, sizeof(v_b[0]),
                                        s_dt, s_ptr, s_stride, 4, assign_error_none);
    EXPECT_TRUE(v_b[0]); EXPECT_FALSE(v_b[1]);
    EXPECT_TRUE(v_b[2]); EXPECT_TRUE(v_b[3]);

    ONE_TEST(int16_type_id, v_i16, 127, 0, -128, -10);
    ONE_TEST(int32_type_id, v_i32, 127, 0, -128, -10);
    ONE_TEST(int64_type_id, v_i64, 127, 0, -128, -10);
    ONE_TEST(uint8_type_id, v_u8, 127u, 0u, (uint8_t)-128, (uint8_t)-10);
    ONE_TEST(uint16_type_id, v_u16, 127u, 0u, (uint16_t)-128, (uint16_t)-10);
    ONE_TEST(uint32_type_id, v_u32, 127u, 0u, (uint32_t)-128, (uint32_t)-10);
    ONE_TEST(uint64_type_id, v_u64, 127u, 0u, (uint64_t)-128, (uint64_t)-10);
    ONE_TEST(float32_type_id, v_f32, 127, 0, -128, -10);
    ONE_TEST(float64_type_id, v_f64, 127, 0, -128, -10);
#undef ONE_TEST
}

TEST(DTypeAssign, FixedSizeTestsStridedNoExcept_Float64) {
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

    s_dt = dtype(float64_type_id);
    s_ptr = v_f64;
    s_stride = 2*sizeof(v_f64[0]);
    v_f64[0] = -10.25; v_f64[1] = 2.25;
    v_f64[2] = 0.0; v_f64[3] = -5.5;
#define ONE_TEST(tid, v, m0, m1) \
            dtype_strided_assign(dtype(tid), v, sizeof(v[0]), s_dt, s_ptr, s_stride, 2, assign_error_none); \
            EXPECT_EQ(m0, v[0]); EXPECT_EQ(m1, v[1])

    dtype_strided_assign(dtype(bool_type_id), v_b, sizeof(v_b[0]),
                                        s_dt, s_ptr, s_stride, 2, assign_error_none);
    EXPECT_TRUE(v_b[0]); EXPECT_FALSE(v_b[1]);

    ONE_TEST(int8_type_id, v_i8, -10, 0);
    ONE_TEST(int16_type_id, v_i16, -10, 0);
    ONE_TEST(int32_type_id, v_i32, -10, 0);
    ONE_TEST(int64_type_id, v_i64, -10, 0);
    ONE_TEST(uint8_type_id, v_u8, (uint8_t)-10, 0u);
    ONE_TEST(uint16_type_id, v_u16, (uint16_t)-10, 0u);
    ONE_TEST(uint32_type_id, v_u32, (uint32_t)-10, 0u);
    ONE_TEST(uint64_type_id, v_u64, (uint64_t)-10, 0u);
    ONE_TEST(float32_type_id, v_f32, -10.25, 0);
#undef ONE_TEST
}


