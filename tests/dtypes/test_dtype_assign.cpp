//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This tests the raw-memory dtype assignment functions.
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <stdint.h>
#include "inc_gtest.hpp"

#include "dnd/dtype.hpp"

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
    complex<float> v_cf32;
    complex<double> v_cf64;

    dtype s_dt, d_dt;
    char *s_ptr;

    // Test bool -> each builtin type
    s_dt = dtype(bool_type_id);
    s_ptr = (char *)&v_b;
    v_b = true;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_none); \
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
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(1));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(1));
#undef ONE_TEST

    // Test int8 -> each builtin type
    s_dt = dtype(int8_type_id);
    s_ptr = (char *)&v_i8;
    v_i8 = 127;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_none); \
            EXPECT_EQ(m, v)
    ONE_TEST(bool_type_id, v_b, true);
    ONE_TEST(int16_type_id, v_i16, 127);
    ONE_TEST(int32_type_id, v_i32, 127);
    ONE_TEST(int64_type_id, v_i64, 127);
    ONE_TEST(uint8_type_id, v_u8, 127u);
    ONE_TEST(uint16_type_id, v_u16, 127u);
    ONE_TEST(uint32_type_id, v_u32, 127u);
    ONE_TEST(uint64_type_id, v_u64, 127u);
    ONE_TEST(float32_type_id, v_f32, 127);
    ONE_TEST(float64_type_id, v_f64, 127);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(127));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(127));
#undef ONE_TEST

    // Test float64 -> each builtin type
    s_dt = dtype(float64_type_id);
    s_ptr = (char *)&v_f64;
    v_f64 = -10.25;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_none); \
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
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(-10.25f));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(-10.25));
#undef ONE_TEST

    // Test complex<float64> -> each builtin type
    s_dt = dtype(complex_float64_type_id);
    s_ptr = (char *)&v_cf64;
    v_cf64 = complex<double>(-10.25, 1.5);
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_none); \
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
    ONE_TEST(float64_type_id, v_f64, -10.25);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(-10.25f, 1.5f));
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
    complex<float> v_cf32;
    complex<double> v_cf64;

    dtype s_dt, d_dt;
    char *s_ptr;

    // Test bool -> each type
    s_dt = dtype(bool_type_id);
    s_ptr = (char *)&v_b;
    v_b = true;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
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
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(1));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(1));
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
    complex<float> v_cf32;
    complex<double> v_cf64;

    dtype s_dt, d_dt;
    char *s_ptr;

    // Test int8 -> types with success
    s_dt = dtype(int8_type_id);
    s_ptr = (char *)&v_i8;
    v_i8 = 127;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
    EXPECT_THROW(dtype_assign(dtype(bool_type_id), (char *)&v_b, s_dt, s_ptr), runtime_error);
    ONE_TEST(int16_type_id, v_i16, 127);
    ONE_TEST(int32_type_id, v_i32, 127);
    ONE_TEST(int64_type_id, v_i64, 127);
    ONE_TEST(uint8_type_id, v_u8, 127u);
    ONE_TEST(uint16_type_id, v_u16, 127u);
    ONE_TEST(uint32_type_id, v_u32, 127u);
    ONE_TEST(uint64_type_id, v_u64, 127u);
    ONE_TEST(float32_type_id, v_f32, 127);
    ONE_TEST(float64_type_id, v_f64, 127);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(127));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(127));
#undef ONE_TEST

    // Test int8 -> bool variants
    v_i8 = -33;
    EXPECT_THROW(dtype_assign(dtype(bool_type_id), (char *)&v_b, s_dt, s_ptr), runtime_error);
    v_i8 = -1;
    EXPECT_THROW(dtype_assign(dtype(bool_type_id), (char *)&v_b, s_dt, s_ptr), runtime_error);
    EXPECT_THROW(dtype_assign(dtype(uint8_type_id), (char *)&v_u8, s_dt, s_ptr), runtime_error);
    v_i8 = 2;
    EXPECT_THROW(dtype_assign(dtype(bool_type_id), (char *)&v_b, s_dt, s_ptr), runtime_error);
    v_i8 = 0;
    dtype_assign(dtype(bool_type_id), (char *)&v_b, s_dt, s_ptr);
    EXPECT_FALSE(v_b);
    v_i8 = 1;
    dtype_assign(dtype(bool_type_id), (char *)&v_b, s_dt, s_ptr);
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
    complex<float> v_cf32;
    complex<double> v_cf64;

    dtype s_dt, d_dt;
    const char *s_ptr;

    s_dt = dtype(float64_type_id);
    s_ptr = reinterpret_cast<char *>(&v_f64);
    v_f64 = -10.25;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_inexact), runtime_error)
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
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(-10.25f));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(-10.25));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // dtype_assign checks that the float64 -> float32value gets converted exactly
    // when using the assign_error_inexact mode
    v_f64 = 1 / 3.0;
    EXPECT_THROW(dtype_assign(dtype(float32_type_id), (char *)&v_f32, s_dt, s_ptr, assign_error_inexact),
                                                                                runtime_error);
    EXPECT_THROW(dtype_assign(dtype(complex_float32_type_id), (char *)&v_cf32, s_dt, s_ptr, assign_error_inexact),
                                                                                runtime_error);
    dtype_assign(dtype(float32_type_id), (char *)&v_f32, s_dt, s_ptr, assign_error_fractional);
    EXPECT_EQ((float)v_f64, v_f32);
    dtype_assign(dtype(complex_float32_type_id), (char *)&v_cf32, s_dt, s_ptr, assign_error_fractional);
    EXPECT_EQ(complex<float>((float)v_f64), v_cf32);

    // Since this is a float -> double conversion, it should be exact coming back to float
    v_f64 = 1 / 3.0f;
    dtype_assign(dtype(float32_type_id), (char *)&v_f32, s_dt, s_ptr, assign_error_inexact);
    EXPECT_EQ(v_f64, v_f32);
    dtype_assign(dtype(complex_float32_type_id), (char *)&v_cf32, s_dt, s_ptr, assign_error_inexact);
    EXPECT_EQ(complex<double>(v_f64), complex<double>(v_cf32));

    // This should overflow converting to float
    v_f64 = -1.5e250;
    EXPECT_THROW(dtype_assign(dtype(float32_type_id), (char *)&v_f32, s_dt, s_ptr, assign_error_inexact),
                                                                                runtime_error);
    EXPECT_THROW(dtype_assign(dtype(float32_type_id), (char *)&v_f32, s_dt, s_ptr, assign_error_fractional),
                                                                                runtime_error);
    EXPECT_THROW(dtype_assign(dtype(float32_type_id), (char *)&v_f32, s_dt, s_ptr, assign_error_overflow),
                                                                                runtime_error);
    EXPECT_THROW(dtype_assign(dtype(complex_float32_type_id), (char *)&v_cf32, s_dt, s_ptr, assign_error_inexact),
                                                                                runtime_error);
    EXPECT_THROW(dtype_assign(dtype(complex_float32_type_id), (char *)&v_cf32, s_dt, s_ptr, assign_error_fractional),
                                                                                runtime_error);
    EXPECT_THROW(dtype_assign(dtype(complex_float32_type_id), (char *)&v_cf32, s_dt, s_ptr, assign_error_overflow),
                                                                                runtime_error);
    dtype_assign(dtype(float32_type_id), (char *)&v_f32, s_dt, s_ptr, assign_error_none);
#ifdef _WIN32
    EXPECT_TRUE(_fpclass(v_f32) == _FPCLASS_NINF);
#else
    EXPECT_TRUE(isinf(v_f32));
#endif
    EXPECT_TRUE(v_f32 < 0);
}

TEST(DTypeAssign, FixedSizeTests_Complex_Float32) {
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
    complex<float> v_cf32;
    complex<double> v_cf64;

    complex<float> v_ref;
    dtype s_dt, d_dt;
    const char *s_ptr;

    s_dt = dtype(complex_float32_type_id);
    s_ptr = reinterpret_cast<char *>(&v_ref);

    // Test the value 0.0
    v_ref = 0.f;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
    dtype_assign(dtype(bool_type_id), (char *)&v_b, s_dt, s_ptr);
    EXPECT_FALSE(v_b);
    ONE_TEST(int8_type_id, v_i8, 0);
    ONE_TEST(int16_type_id, v_i16, 0);
    ONE_TEST(int32_type_id, v_i32, 0);
    ONE_TEST(int64_type_id, v_i64, 0);
    ONE_TEST(uint8_type_id, v_u8, 0u);
    ONE_TEST(uint16_type_id, v_u16, 0u);
    ONE_TEST(uint32_type_id, v_u32, 0u);
    ONE_TEST(uint64_type_id, v_u64, 0u);
    ONE_TEST(float32_type_id, v_f32, 0);
    ONE_TEST(float64_type_id, v_f64, 0);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(0));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(0));
#undef ONE_TEST

    // Test the value 1.0
    v_ref = 1.f;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
    dtype_assign(dtype(bool_type_id), (char *)&v_b, s_dt, s_ptr);
    EXPECT_TRUE(v_b);
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
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(1));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(1));
#undef ONE_TEST

    // Test the value 2.0
    v_ref = 2.f;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_inexact), runtime_error)
    ONE_TEST_THROW(bool_type_id, v_b);
    ONE_TEST(int8_type_id, v_i8, 2);
    ONE_TEST(int16_type_id, v_i16, 2);
    ONE_TEST(int32_type_id, v_i32, 2);
    ONE_TEST(int64_type_id, v_i64, 2);
    ONE_TEST(uint8_type_id, v_u8, 2u);
    ONE_TEST(uint16_type_id, v_u16, 2u);
    ONE_TEST(uint32_type_id, v_u32, 2u);
    ONE_TEST(uint64_type_id, v_u64, 2u);
    ONE_TEST(float32_type_id, v_f32, 2);
    ONE_TEST(float64_type_id, v_f64, 2);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(2));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(2));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test the value -1.0
    v_ref = -1.f;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_inexact), runtime_error)
    ONE_TEST_THROW(bool_type_id, v_b);
    ONE_TEST(int8_type_id, v_i8, -1);
    ONE_TEST(int16_type_id, v_i16, -1);
    ONE_TEST(int32_type_id, v_i32, -1);
    ONE_TEST(int64_type_id, v_i64, -1);
    ONE_TEST_THROW(uint8_type_id, v_b);
    ONE_TEST_THROW(uint16_type_id, v_b);
    ONE_TEST_THROW(uint32_type_id, v_b);
    ONE_TEST_THROW(uint64_type_id, v_b);
    ONE_TEST(float32_type_id, v_f32, -1);
    ONE_TEST(float64_type_id, v_f64, -1);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(-1));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(-1));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test a real value stored in complex
    v_ref = -10.25f;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_inexact), runtime_error)
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
    ONE_TEST(float64_type_id, v_f64, -10.25);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(-10.25f));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(-10.25));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test a large integer value stored in complex
    v_ref = 1e21f;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_inexact), runtime_error)
    ONE_TEST_THROW(bool_type_id, v_b);
    ONE_TEST_THROW(int8_type_id, v_i8);
    ONE_TEST_THROW(int16_type_id, v_i16);
    ONE_TEST_THROW(int32_type_id, v_i32);
    ONE_TEST_THROW(int64_type_id, v_i64);
    ONE_TEST_THROW(uint8_type_id, v_u8);
    ONE_TEST_THROW(uint16_type_id, v_u16);
    ONE_TEST_THROW(uint32_type_id, v_u32);
    ONE_TEST_THROW(uint64_type_id, v_u64);
    ONE_TEST(float32_type_id, v_f32, 1e21f);
    ONE_TEST(float64_type_id, v_f64, 1e21f);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(1e21f));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(1e21f));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test a complex value with imaginary component
    v_ref = complex<float>(-10.25f, 0.125f);
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), &v, s_dt, s_ptr, assign_error_inexact), runtime_error)
    ONE_TEST_THROW(bool_type_id, v_b);
    ONE_TEST_THROW(int8_type_id, v_i8);
    ONE_TEST_THROW(int16_type_id, v_i16);
    ONE_TEST_THROW(int32_type_id, v_i32);
    ONE_TEST_THROW(int64_type_id, v_i64);
    ONE_TEST_THROW(uint8_type_id, v_u8);
    ONE_TEST_THROW(uint16_type_id, v_u16);
    ONE_TEST_THROW(uint32_type_id, v_u32);
    ONE_TEST_THROW(uint64_type_id, v_u64);
    ONE_TEST_THROW(float32_type_id, v_f32);
    ONE_TEST_THROW(float64_type_id, v_f64);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(-10.25f,0.125f));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(-10.25,0.125));
#undef ONE_TEST
#undef ONE_TEST_THROW
}

TEST(DTypeAssign, FixedSizeTests_Complex_Float64) {
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
    complex<float> v_cf32;
    complex<double> v_cf64;

    complex<double> v_ref;
    dtype s_dt, d_dt;
    const char *s_ptr;

    s_dt = dtype(complex_float64_type_id);
    s_ptr = reinterpret_cast<char *>(&v_ref);

    // Test the value 0.0
    v_ref = 0.;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
    dtype_assign(dtype(bool_type_id), (char *)&v_b, s_dt, s_ptr);
    EXPECT_FALSE(v_b);
    ONE_TEST(int8_type_id, v_i8, 0);
    ONE_TEST(int16_type_id, v_i16, 0);
    ONE_TEST(int32_type_id, v_i32, 0);
    ONE_TEST(int64_type_id, v_i64, 0);
    ONE_TEST(uint8_type_id, v_u8, 0u);
    ONE_TEST(uint16_type_id, v_u16, 0u);
    ONE_TEST(uint32_type_id, v_u32, 0u);
    ONE_TEST(uint64_type_id, v_u64, 0u);
    ONE_TEST(float32_type_id, v_f32, 0);
    ONE_TEST(float64_type_id, v_f64, 0);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(0));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(0));
#undef ONE_TEST

    // Test the value 1.0
    v_ref = 1.;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
    dtype_assign(dtype(bool_type_id), (char *)&v_b, s_dt, s_ptr);
    EXPECT_TRUE(v_b);
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
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(1));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(1));
#undef ONE_TEST

    // Test the value 2.0
    v_ref = 2.;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_inexact), runtime_error)
    ONE_TEST_THROW(bool_type_id, v_b);
    ONE_TEST(int8_type_id, v_i8, 2);
    ONE_TEST(int16_type_id, v_i16, 2);
    ONE_TEST(int32_type_id, v_i32, 2);
    ONE_TEST(int64_type_id, v_i64, 2);
    ONE_TEST(uint8_type_id, v_u8, 2u);
    ONE_TEST(uint16_type_id, v_u16, 2u);
    ONE_TEST(uint32_type_id, v_u32, 2u);
    ONE_TEST(uint64_type_id, v_u64, 2u);
    ONE_TEST(float32_type_id, v_f32, 2);
    ONE_TEST(float64_type_id, v_f64, 2);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(2));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(2));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test the value -1.0
    v_ref = -1.;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_inexact), runtime_error)
    ONE_TEST_THROW(bool_type_id, v_b);
    ONE_TEST(int8_type_id, v_i8, -1);
    ONE_TEST(int16_type_id, v_i16, -1);
    ONE_TEST(int32_type_id, v_i32, -1);
    ONE_TEST(int64_type_id, v_i64, -1);
    ONE_TEST_THROW(uint8_type_id, v_b);
    ONE_TEST_THROW(uint16_type_id, v_b);
    ONE_TEST_THROW(uint32_type_id, v_b);
    ONE_TEST_THROW(uint64_type_id, v_b);
    ONE_TEST(float32_type_id, v_f32, -1);
    ONE_TEST(float64_type_id, v_f64, -1);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(-1));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(-1));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test a real value stored in complex
    v_ref = -10.25;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_inexact), runtime_error)
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
    ONE_TEST(float64_type_id, v_f64, -10.25);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(-10.25f));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(-10.25));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test a large integer value stored in complex
    v_ref = 1e21;
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_inexact), runtime_error)
    ONE_TEST_THROW(bool_type_id, v_b);
    ONE_TEST_THROW(int8_type_id, v_i8);
    ONE_TEST_THROW(int16_type_id, v_i16);
    ONE_TEST_THROW(int32_type_id, v_i32);
    ONE_TEST_THROW(int64_type_id, v_i64);
    ONE_TEST_THROW(uint8_type_id, v_u8);
    ONE_TEST_THROW(uint16_type_id, v_u16);
    ONE_TEST_THROW(uint32_type_id, v_u32);
    ONE_TEST_THROW(uint64_type_id, v_u64);
    ONE_TEST(float32_type_id, v_f32, float(1e21));
    ONE_TEST(float64_type_id, v_f64, 1e21);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(float(1e21)));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(1e21));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test a complex value with imaginary component
    v_ref = complex<double>(-10.25, 0.125);
#define ONE_TEST(tid, v, m) \
            dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(dtype_assign(dtype(tid), (char *)&v, s_dt, s_ptr, assign_error_inexact), runtime_error)
    ONE_TEST_THROW(bool_type_id, v_b);
    ONE_TEST_THROW(int8_type_id, v_i8);
    ONE_TEST_THROW(int16_type_id, v_i16);
    ONE_TEST_THROW(int32_type_id, v_i32);
    ONE_TEST_THROW(int64_type_id, v_i64);
    ONE_TEST_THROW(uint8_type_id, v_u8);
    ONE_TEST_THROW(uint16_type_id, v_u16);
    ONE_TEST_THROW(uint32_type_id, v_u32);
    ONE_TEST_THROW(uint64_type_id, v_u64);
    ONE_TEST_THROW(float32_type_id, v_f32);
    ONE_TEST_THROW(float64_type_id, v_f64);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(-10.25f,0.125f));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(-10.25,0.125));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // dtype_assign checks that the float64 -> float32value gets converted exactly
    // when using the assign_error_inexact mode
    v_cf64 = 1 / 3.0;
    v_f32 = 0.f;
    EXPECT_THROW(dtype_assign(dtype(float32_type_id), (char *)&v_f32, dtype(complex_float64_type_id), (char *)&v_cf64, assign_error_inexact),
                                                                                runtime_error);
    v_cf32 = 0.f;
    EXPECT_THROW(dtype_assign(dtype(complex_float32_type_id), (char *)&v_cf32, dtype(complex_float64_type_id), (char *)&v_cf64, assign_error_inexact),
                                                                                runtime_error);
    dtype_assign(dtype(float32_type_id), (char *)&v_f32, dtype(complex_float64_type_id), (char *)&v_cf64, assign_error_fractional);
    EXPECT_EQ((float)v_cf64.real(), v_f32);
    dtype_assign(dtype(complex_float32_type_id), (char *)&v_cf32, dtype(complex_float64_type_id), (char *)&v_cf64, assign_error_fractional);
    EXPECT_EQ(complex<float>((float)v_cf64.real()), v_cf32);

    // Since this is a float -> double conversion, it should be exact coming back to float
    v_cf64 = 1 / 3.0f;
    dtype_assign(dtype(float32_type_id), (char *)&v_f32, dtype(complex_float64_type_id), (char *)&v_cf64, assign_error_inexact);
    EXPECT_EQ(v_cf64, complex<double>(v_f32));
    dtype_assign(dtype(complex_float32_type_id), (char *)&v_cf32, dtype(complex_float64_type_id), (char *)&v_cf64, assign_error_inexact);
    EXPECT_EQ(v_cf64, complex<double>(v_cf32));

    // This should overflow converting to float
    v_cf64 = -1.5e250;
    EXPECT_THROW(dtype_assign(dtype(float32_type_id), (char *)&v_f32, dtype(complex_float64_type_id), (char *)&v_cf64, assign_error_inexact),
                                                                                runtime_error);
    EXPECT_THROW(dtype_assign(dtype(float32_type_id), (char *)&v_f32, dtype(complex_float64_type_id), (char *)&v_cf64, assign_error_fractional),
                                                                                runtime_error);
    EXPECT_THROW(dtype_assign(dtype(float32_type_id), (char *)&v_f32, dtype(complex_float64_type_id), (char *)&v_cf64, assign_error_overflow),
                                                                                runtime_error);
    EXPECT_THROW(dtype_assign(dtype(complex_float32_type_id), (char *)&v_cf32, dtype(complex_float64_type_id), (char *)&v_cf64, assign_error_inexact),
                                                                                runtime_error);
    EXPECT_THROW(dtype_assign(dtype(complex_float32_type_id), (char *)&v_cf32, dtype(complex_float64_type_id), (char *)&v_cf64, assign_error_fractional),
                                                                                runtime_error);
    EXPECT_THROW(dtype_assign(dtype(complex_float32_type_id), (char *)&v_cf32, dtype(complex_float64_type_id), (char *)&v_cf64, assign_error_overflow),
                                                                                runtime_error);
    dtype_assign(dtype(float32_type_id), (char *)&v_f32, dtype(complex_float64_type_id), (char *)&v_cf64, assign_error_none);
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
    complex<float> v_cf32[4];
    complex<double> v_cf64[4];

    dtype s_dt, d_dt;
    char *s_ptr;
    intptr_t s_stride;

    s_dt = dtype(bool_type_id);
    s_ptr = (char *)v_b;
    s_stride = sizeof(v_b[0]);
    v_b[0] = true; v_b[1] = true; v_b[2] = false; v_b[3] = true;
#define ONE_TEST(type, v) \
            dtype_strided_assign(make_dtype<type>(), (char *)v, sizeof(v[0]), s_dt, s_ptr, s_stride, 4, assign_error_none); \
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
    ONE_TEST(complex<float>, v_cf32);
    ONE_TEST(complex<double>, v_cf64);
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
    complex<float> v_cf32[4];
    complex<double> v_cf64[4];

    dtype s_dt, d_dt;
    char *s_ptr;
    intptr_t s_stride;

    s_dt = dtype(int8_type_id);
    s_ptr = (char *)v_i8;
    s_stride = sizeof(v_i8[0]);
    v_i8[0] = 127; v_i8[1] = 0; v_i8[2] = -128; v_i8[3] = -10;
#define ONE_TEST(tid, v, m0, m1, m2, m3) \
            dtype_strided_assign(dtype(tid), (char *)v, sizeof(v[0]), s_dt, s_ptr, s_stride, 4, assign_error_none); \
            EXPECT_EQ(m0, v[0]); EXPECT_EQ(m1, v[1]); \
            EXPECT_EQ(m2, v[2]); EXPECT_EQ(m3, v[3])

    dtype_strided_assign(dtype(bool_type_id), (char *)v_b, sizeof(v_b[0]),
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
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(127), complex<float>(0), complex<float>(-128), complex<float>(-10));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(127), complex<double>(0), complex<double>(-128), complex<double>(-10));
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
    complex<float> v_cf32[4];
    complex<double> v_cf64[4];

    dtype s_dt, d_dt;
    char *s_ptr;
    intptr_t s_stride;

    s_dt = dtype(float64_type_id);
    s_ptr = (char *)v_f64;
    s_stride = 2*sizeof(v_f64[0]);
    v_f64[0] = -10.25; v_f64[1] = 2.25;
    v_f64[2] = 0.0; v_f64[3] = -5.5;
#define ONE_TEST(tid, v, m0, m1) \
            dtype_strided_assign(dtype(tid), (char *)v, sizeof(v[0]), s_dt, s_ptr, s_stride, 2, assign_error_none); \
            EXPECT_EQ(m0, v[0]); EXPECT_EQ(m1, v[1])

    dtype_strided_assign(dtype(bool_type_id), (char *)v_b, sizeof(v_b[0]),
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
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(-10.25f), complex<float>(0));
    ONE_TEST(complex_float64_type_id, v_cf64, complex<double>(-10.25), complex<double>(0));
#undef ONE_TEST
}

TEST(DTypeAssign, FixedSizeTestsStridedNoExcept_Complex_Float64) {
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
    complex<float> v_cf32[4];
    complex<double> v_cf64[4];

    dtype s_dt, d_dt;
    char *s_ptr;
    intptr_t s_stride;

    s_dt = dtype(complex_float64_type_id);
    s_ptr = (char *)v_cf64;
    s_stride = 2*sizeof(v_cf64[0]);
    v_cf64[0] = complex<double>(-10.25, 1.5); v_cf64[1] = complex<double>(2.25, -3.125);
    v_cf64[2] = 0.0; v_cf64[3] = complex<double>(0,1.5);
#define ONE_TEST(tid, v, m0, m1) \
            dtype_strided_assign(dtype(tid), (char *)v, sizeof(v[0]), s_dt, s_ptr, s_stride, 2, assign_error_none); \
            EXPECT_EQ(m0, v[0]); EXPECT_EQ(m1, v[1])

    dtype_strided_assign(dtype(bool_type_id), (char *)v_b, sizeof(v_b[0]),
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
    ONE_TEST(float32_type_id, v_f32, -10.25f, 0);
    ONE_TEST(float64_type_id, v_f64, -10.25, 0);
    ONE_TEST(complex_float32_type_id, v_cf32, complex<float>(-10.25f, 1.5), complex<float>(0));
#undef ONE_TEST
}



