//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This tests the raw-memory typed data assignment functions.
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include "inc_gtest.hpp"

#include "dynd/type.hpp"

using namespace std;
using namespace dynd;

TEST(DTypeAssign, FixedSizeTestsNoExcept) {
    dynd_bool v_b;
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
    dynd_complex<float> v_cf32;
    dynd_complex<double> v_cf64;

    ndt::type s_dt, d_dt;
    char *s_ptr;

    // Test bool -> each builtin type
    s_dt = ndt::type(bool_type_id);
    s_ptr = (char *)&v_b;
    v_b = true;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr, assign_error_none); \
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(1));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(1));
#undef ONE_TEST

    // Test int8 -> each builtin type
    s_dt = ndt::type(int8_type_id);
    s_ptr = (char *)&v_i8;
    v_i8 = 127;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr, assign_error_none); \
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(127));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(127));
#undef ONE_TEST

    // Test float64 -> each builtin type
    s_dt = ndt::type(float64_type_id);
    s_ptr = (char *)&v_f64;
    v_f64 = -10.25;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr, assign_error_none); \
            EXPECT_EQ(m, v)
    ONE_TEST(bool_type_id, v_b, true);
    ONE_TEST(int8_type_id, v_i8, -10);
    ONE_TEST(int16_type_id, v_i16, -10);
    ONE_TEST(int32_type_id, v_i32, -10);
    ONE_TEST(int64_type_id, v_i64, -10);
    ONE_TEST(uint8_type_id, v_u8, (uint8_t)-10);
    ONE_TEST(uint16_type_id, v_u16, (uint16_t)-10);
    ONE_TEST(uint32_type_id, v_u32, (uint32_t)-10);
    // This float64 -> uint64 if commented out because it
    // behaves differently on linux 32. The behavior is
    // not well-defined according to C/C++, so that should
    // be ok.
    //ONE_TEST(uint64_type_id, v_u64, (uint64_t)-10);
    ONE_TEST(float32_type_id, v_f32, -10.25);
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(-10.25f));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(-10.25));
#undef ONE_TEST

    // Test dynd_complex<float64> -> each builtin type
    s_dt = ndt::type(complex_float64_type_id);
    s_ptr = (char *)&v_cf64;
    v_cf64 = dynd_complex<double>(-10.25, 1.5);
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr, assign_error_none); \
            EXPECT_EQ(m, v)
    ONE_TEST(bool_type_id, v_b, true);
    ONE_TEST(int8_type_id, v_i8, -10);
    ONE_TEST(int16_type_id, v_i16, -10);
    ONE_TEST(int32_type_id, v_i32, -10);
    ONE_TEST(int64_type_id, v_i64, -10);
    ONE_TEST(uint8_type_id, v_u8, (uint8_t)-10);
    ONE_TEST(uint16_type_id, v_u16, (uint16_t)-10);
    ONE_TEST(uint32_type_id, v_u32, (uint32_t)-10);
    // This float64 -> uint64 if commented out because it
    // behaves differently on linux 32. The behavior is
    // not well-defined according to C/C++, so that should
    // be ok.
    //ONE_TEST(uint64_type_id, v_u64, (uint64_t)-10);
    ONE_TEST(float32_type_id, v_f32, -10.25);
    ONE_TEST(float64_type_id, v_f64, -10.25);
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(-10.25f, 1.5f));
#undef ONE_TEST
}

TEST(DTypeAssign, FixedSizeTests_Bool) {
    dynd_bool v_b;
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
    dynd_complex<float> v_cf32;
    dynd_complex<double> v_cf64;

    ndt::type s_dt, d_dt;
    char *s_ptr;

    // Test bool -> each type
    s_dt = ndt::type(bool_type_id);
    s_ptr = (char *)&v_b;
    v_b = true;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(1));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(1));
#undef ONE_TEST
}

TEST(DTypeAssign, FixedSizeTests_Int8) {
    dynd_bool v_b;
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
    dynd_complex<float> v_cf32;
    dynd_complex<double> v_cf64;

    ndt::type s_dt, d_dt;
    char *s_ptr;

    // Test int8 -> types with success
    s_dt = ndt::type(int8_type_id);
    s_ptr = (char *)&v_i8;
    v_i8 = 127;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
    EXPECT_THROW(typed_data_assign(ndt::type(bool_type_id), NULL, (char *)&v_b, s_dt, NULL, s_ptr), runtime_error);
    ONE_TEST(int16_type_id, v_i16, 127);
    ONE_TEST(int32_type_id, v_i32, 127);
    ONE_TEST(int64_type_id, v_i64, 127);
    ONE_TEST(uint8_type_id, v_u8, 127u);
    ONE_TEST(uint16_type_id, v_u16, 127u);
    ONE_TEST(uint32_type_id, v_u32, 127u);
    ONE_TEST(uint64_type_id, v_u64, 127u);
    ONE_TEST(float32_type_id, v_f32, 127);
    ONE_TEST(float64_type_id, v_f64, 127);
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(127));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(127));
#undef ONE_TEST

    // Test int8 -> bool variants
    v_i8 = -33;
    EXPECT_THROW(typed_data_assign(ndt::type(bool_type_id), NULL, (char *)&v_b, s_dt, NULL, s_ptr), runtime_error);
    v_i8 = -1;
    EXPECT_THROW(typed_data_assign(ndt::type(bool_type_id), NULL, (char *)&v_b, s_dt, NULL, s_ptr), runtime_error);
    EXPECT_THROW(typed_data_assign(ndt::type(uint8_type_id), NULL, (char *)&v_u8, s_dt, NULL, s_ptr), runtime_error);
    v_i8 = 2;
    EXPECT_THROW(typed_data_assign(ndt::type(bool_type_id), NULL, (char *)&v_b, s_dt, NULL, s_ptr), runtime_error);
    v_i8 = 0;
    typed_data_assign(ndt::type(bool_type_id), NULL, (char *)&v_b, s_dt, NULL, s_ptr);
    EXPECT_FALSE(v_b);
    v_i8 = 1;
    typed_data_assign(ndt::type(bool_type_id), NULL, (char *)&v_b, s_dt, NULL, s_ptr);
    EXPECT_TRUE(v_b);
}

TEST(DTypeAssign, FixedSizeTests_Float64) {
    dynd_bool v_b;
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
    dynd_complex<float> v_cf32;
    dynd_complex<double> v_cf64;

    ndt::type s_dt, d_dt;
    const char *s_ptr;

    s_dt = ndt::type(float64_type_id);
    s_ptr = reinterpret_cast<char *>(&v_f64);
    v_f64 = -10.25;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_inexact), runtime_error)
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(-10.25f));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(-10.25));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // typed_data_assign checks that the float64 -> float32value gets converted exactly
    // when using the assign_error_inexact mode
    v_f64 = 1 / 3.0;
    EXPECT_THROW(typed_data_assign(ndt::type(float32_type_id), NULL, (char *)&v_f32, s_dt, NULL, s_ptr, assign_error_inexact),
                                                                                runtime_error);
    EXPECT_THROW(typed_data_assign(ndt::type(complex_float32_type_id), NULL, (char *)&v_cf32, s_dt, NULL, s_ptr, assign_error_inexact),
                                                                                runtime_error);
    typed_data_assign(ndt::type(float32_type_id), NULL, (char *)&v_f32, s_dt, NULL, s_ptr, assign_error_fractional);
    EXPECT_EQ((float)v_f64, v_f32);
    typed_data_assign(ndt::type(complex_float32_type_id), NULL, (char *)&v_cf32, s_dt, NULL, s_ptr, assign_error_fractional);
    EXPECT_EQ(dynd_complex<float>((float)v_f64), v_cf32);

    // Since this is a float -> double conversion, it should be exact coming back to float
    v_f64 = 1 / 3.0f;
    typed_data_assign(ndt::type(float32_type_id), NULL, (char *)&v_f32, s_dt, NULL, s_ptr, assign_error_inexact);
    EXPECT_EQ(v_f64, v_f32);
    typed_data_assign(ndt::type(complex_float32_type_id), NULL, (char *)&v_cf32, s_dt, NULL, s_ptr, assign_error_inexact);
    EXPECT_EQ(dynd_complex<double>(v_f64), dynd_complex<double>(v_cf32));

    // This should overflow converting to float
    v_f64 = -1.5e250;
    EXPECT_THROW(typed_data_assign(ndt::type(float32_type_id), NULL, (char *)&v_f32, s_dt, NULL, s_ptr, assign_error_inexact),
                                                                                runtime_error);
    EXPECT_THROW(typed_data_assign(ndt::type(float32_type_id), NULL, (char *)&v_f32, s_dt, NULL, s_ptr, assign_error_fractional),
                                                                                runtime_error);
    EXPECT_THROW(typed_data_assign(ndt::type(float32_type_id), NULL, (char *)&v_f32, s_dt, NULL, s_ptr, assign_error_overflow),
                                                                                runtime_error);
    EXPECT_THROW(typed_data_assign(ndt::type(complex_float32_type_id), NULL, (char *)&v_cf32, s_dt, NULL, s_ptr, assign_error_inexact),
                                                                                runtime_error);
    EXPECT_THROW(typed_data_assign(ndt::type(complex_float32_type_id), NULL, (char *)&v_cf32, s_dt, NULL, s_ptr, assign_error_fractional),
                                                                                runtime_error);
    EXPECT_THROW(typed_data_assign(ndt::type(complex_float32_type_id), NULL, (char *)&v_cf32, s_dt, NULL, s_ptr, assign_error_overflow),
                                                                                runtime_error);
    typed_data_assign(ndt::type(float32_type_id), NULL, (char *)&v_f32, s_dt, NULL, s_ptr, assign_error_none);
#ifdef _WIN32
    EXPECT_TRUE(_fpclass(v_f32) == _FPCLASS_NINF);
#else
    EXPECT_TRUE(isinf(v_f32));
#endif
    EXPECT_TRUE(v_f32 < 0);
}

TEST(DTypeAssign, FixedSizeTests_Complex_Float32) {
    dynd_bool v_b;
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
    dynd_complex<float> v_cf32;
    dynd_complex<double> v_cf64;

    dynd_complex<float> v_ref;
    ndt::type s_dt, d_dt;
    const char *s_ptr;

    s_dt = ndt::type(complex_float32_type_id);
    s_ptr = reinterpret_cast<char *>(&v_ref);

    // Test the value 0.0
    v_ref = 0.f;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
    typed_data_assign(ndt::type(bool_type_id), NULL, (char *)&v_b, s_dt, NULL, s_ptr);
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(0));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(0));
#undef ONE_TEST

    // Test the value 1.0
    v_ref = 1.f;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
    typed_data_assign(ndt::type(bool_type_id), NULL, (char *)&v_b, s_dt, NULL, s_ptr);
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(1));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(1));
#undef ONE_TEST

    // Test the value 2.0
    v_ref = 2.f;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), &v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), &v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), &v, s_dt, s_ptr, assign_error_inexact), runtime_error)
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(2));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(2));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test the value -1.0
    v_ref = -1.f;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), &v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), &v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), &v, s_dt, s_ptr, assign_error_inexact), runtime_error)
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(-1));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(-1));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test a real value stored in complex
    v_ref = -10.25f;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), &v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), &v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), &v, s_dt, s_ptr, assign_error_inexact), runtime_error)
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(-10.25f));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(-10.25));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test a large integer value stored in complex
    v_ref = 1e21f;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), &v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), &v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), &v, s_dt, s_ptr, assign_error_inexact), runtime_error)
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(1e21f));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(1e21f));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test a complex value with imaginary component
    v_ref = dynd_complex<float>(-10.25f, 0.125f);
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), &v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), &v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), &v, s_dt, s_ptr, assign_error_inexact), runtime_error)
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(-10.25f,0.125f));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(-10.25,0.125));
#undef ONE_TEST
#undef ONE_TEST_THROW
}

TEST(DTypeAssign, FixedSizeTests_Complex_Float64) {
    dynd_bool v_b;
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
    dynd_complex<float> v_cf32;
    dynd_complex<double> v_cf64;

    dynd_complex<double> v_ref;
    ndt::type s_dt, d_dt;
    const char *s_ptr;

    s_dt = ndt::type(complex_float64_type_id);
    s_ptr = reinterpret_cast<char *>(&v_ref);

    // Test the value 0.0
    v_ref = 0.;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
    typed_data_assign(ndt::type(bool_type_id), NULL, (char *)&v_b, s_dt, NULL, s_ptr);
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(0));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(0));
#undef ONE_TEST

    // Test the value 1.0
    v_ref = 1.;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
    typed_data_assign(ndt::type(bool_type_id), NULL, (char *)&v_b, s_dt, NULL, s_ptr);
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(1));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(1));
#undef ONE_TEST

    // Test the value 2.0
    v_ref = 2.;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_inexact), runtime_error)
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(2));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(2));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test the value -1.0
    v_ref = -1.;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_inexact), runtime_error)
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(-1));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(-1));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test a real value stored in complex
    v_ref = -10.25;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_inexact), runtime_error)
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(-10.25f));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(-10.25));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test a large integer value stored in complex
    v_ref = 1e21;
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_inexact), runtime_error)
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(float(1e21)));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(1e21));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // Test a complex value with imaginary component
    v_ref = dynd_complex<double>(-10.25, 0.125);
#define ONE_TEST(tid, v, m) \
            typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr); \
            EXPECT_EQ(m, v)
#define ONE_TEST_THROW(tid, v) \
            EXPECT_THROW(typed_data_assign(ndt::type(tid), NULL, (char *)&v, s_dt, NULL, s_ptr), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_overflow), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_fractional), runtime_error);
//            EXPECT_THROW(typed_data_assign(ndt::type(tid), (char *)&v, s_dt, s_ptr, assign_error_inexact), runtime_error)
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
    ONE_TEST(complex_float32_type_id, v_cf32, dynd_complex<float>(-10.25f,0.125f));
    ONE_TEST(complex_float64_type_id, v_cf64, dynd_complex<double>(-10.25,0.125));
#undef ONE_TEST
#undef ONE_TEST_THROW

    // typed_data_assign checks that the float64 -> float32value gets converted exactly
    // when using the assign_error_inexact mode
    v_cf64 = 1 / 3.0;
    v_f32 = 0.f;
    EXPECT_THROW(typed_data_assign(ndt::type(float32_type_id), NULL, (char *)&v_f32, ndt::type(complex_float64_type_id), NULL, (char *)&v_cf64, assign_error_inexact),
                                                                                runtime_error);
    v_cf32 = 0.f;
    EXPECT_THROW(typed_data_assign(ndt::type(complex_float32_type_id), NULL, (char *)&v_cf32, ndt::type(complex_float64_type_id), NULL, (char *)&v_cf64, assign_error_inexact),
                                                                                runtime_error);
    typed_data_assign(ndt::type(float32_type_id), NULL, (char *)&v_f32, ndt::type(complex_float64_type_id), NULL, (char *)&v_cf64, assign_error_fractional);
    EXPECT_EQ((float)v_cf64.real(), v_f32);
    typed_data_assign(ndt::type(complex_float32_type_id), NULL, (char *)&v_cf32, ndt::type(complex_float64_type_id), NULL, (char *)&v_cf64, assign_error_fractional);
    EXPECT_EQ(dynd_complex<float>((float)v_cf64.real()), v_cf32);

    // Since this is a float -> double conversion, it should be exact coming back to float
    v_cf64 = 1 / 3.0f;
    typed_data_assign(ndt::type(float32_type_id), NULL, (char *)&v_f32, ndt::type(complex_float64_type_id), NULL, (char *)&v_cf64, assign_error_inexact);
    EXPECT_EQ(v_cf64, dynd_complex<double>(v_f32));
    typed_data_assign(ndt::type(complex_float32_type_id), NULL, (char *)&v_cf32, ndt::type(complex_float64_type_id), NULL, (char *)&v_cf64, assign_error_inexact);
    EXPECT_EQ(v_cf64, dynd_complex<double>(v_cf32));

    // This should overflow converting to float
    v_cf64 = -1.5e250;
    EXPECT_THROW(typed_data_assign(ndt::type(float32_type_id), NULL, (char *)&v_f32, ndt::type(complex_float64_type_id), NULL, (char *)&v_cf64, assign_error_inexact),
                                                                                runtime_error);
    EXPECT_THROW(typed_data_assign(ndt::type(float32_type_id), NULL, (char *)&v_f32, ndt::type(complex_float64_type_id), NULL, (char *)&v_cf64, assign_error_fractional),
                                                                                runtime_error);
    EXPECT_THROW(typed_data_assign(ndt::type(float32_type_id), NULL, (char *)&v_f32, ndt::type(complex_float64_type_id), NULL, (char *)&v_cf64, assign_error_overflow),
                                                                                runtime_error);
    EXPECT_THROW(typed_data_assign(ndt::type(complex_float32_type_id), NULL, (char *)&v_cf32, ndt::type(complex_float64_type_id), NULL, (char *)&v_cf64, assign_error_inexact),
                                                                                runtime_error);
    EXPECT_THROW(typed_data_assign(ndt::type(complex_float32_type_id), NULL, (char *)&v_cf32, ndt::type(complex_float64_type_id), NULL, (char *)&v_cf64, assign_error_fractional),
                                                                                runtime_error);
    EXPECT_THROW(typed_data_assign(ndt::type(complex_float32_type_id), NULL, (char *)&v_cf32, ndt::type(complex_float64_type_id), NULL, (char *)&v_cf64, assign_error_overflow),
                                                                                runtime_error);
    typed_data_assign(ndt::type(float32_type_id), NULL, (char *)&v_f32, ndt::type(complex_float64_type_id), NULL, (char *)&v_cf64, assign_error_none);
#ifdef _WIN32
    EXPECT_TRUE(_fpclass(v_f32) == _FPCLASS_NINF);
#else
    EXPECT_TRUE(isinf(v_f32));
#endif
    EXPECT_TRUE(v_f32 < 0);

}

