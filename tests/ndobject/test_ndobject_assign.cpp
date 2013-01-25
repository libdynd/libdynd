//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include "inc_gtest.hpp"

#include <dynd/ndobject.hpp>
#include <dynd/dtypes/convert_dtype.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>

using namespace std;
using namespace dynd;

TEST(NDObjectAssign, ScalarAssignment_Bool) {
    ndobject a;

    // assignment to a bool scalar
    a = ndobject(make_dtype<dynd_bool>());
    const dynd_bool *ptr_b = (const dynd_bool *)a.get_ndo()->m_data_pointer;
    a.val_assign(true);
    EXPECT_TRUE(*ptr_b);
    a.val_assign(false);
    EXPECT_FALSE(*ptr_b);
    a.val_assign(1);
    EXPECT_TRUE(*ptr_b);
    a.val_assign(0);
    EXPECT_FALSE(*ptr_b);
    a.val_assign(1.0);
    EXPECT_TRUE(*ptr_b);
    a.val_assign(0.0);
    EXPECT_FALSE(*ptr_b);
    a.val_assign(1.5, assign_error_none);
    EXPECT_TRUE(*ptr_b);
    a.val_assign(-3.5f, assign_error_none);
    EXPECT_TRUE(*ptr_b);
    a.val_assign(22, assign_error_none);
    EXPECT_TRUE(*ptr_b);
    EXPECT_THROW(a.val_assign(2), runtime_error);
    EXPECT_THROW(a.val_assign(-1), runtime_error);
    EXPECT_THROW(a.val_assign(1.5), runtime_error);
    EXPECT_THROW(a.val_assign(1.5, assign_error_overflow), runtime_error);
    EXPECT_THROW(a.val_assign(1.5, assign_error_fractional), runtime_error);
    EXPECT_THROW(a.val_assign(1.5, assign_error_inexact), runtime_error);
}

TEST(NDObjectAssign, ScalarAssignment_Int8) {
    ndobject a;
    const int8_t *ptr_i8;

    // Assignment to an int8_t scalar
    a = ndobject(make_dtype<int8_t>());
    ptr_i8 = (const int8_t *)a.get_ndo()->m_data_pointer;
    a.val_assign(true);
    EXPECT_EQ(1, *ptr_i8);
    a.val_assign(false);
    EXPECT_EQ(0, *ptr_i8);
    a.val_assign(-10);
    EXPECT_EQ(-10, *ptr_i8);
    a.val_assign(-128);
    EXPECT_EQ(-128, *ptr_i8);
    a.val_assign(127);
    EXPECT_EQ(127, *ptr_i8);
    EXPECT_THROW(a.val_assign(-129), runtime_error);
    EXPECT_THROW(a.val_assign(128), runtime_error);
    a.val_assign(5.0);
    EXPECT_EQ(5, *ptr_i8);
    a.val_assign(-100.0f);
    EXPECT_EQ(-100, *ptr_i8);
    EXPECT_THROW(a.val_assign(1.25), runtime_error);
    EXPECT_THROW(a.val_assign(128.0), runtime_error);
    EXPECT_THROW(a.val_assign(128.0, assign_error_inexact), runtime_error);
    EXPECT_THROW(a.val_assign(1e30), runtime_error);
    a.val_assign(1.25, assign_error_overflow);
    EXPECT_EQ(1, *ptr_i8);
    EXPECT_THROW(a.val_assign(-129.0, assign_error_overflow), runtime_error);
    a.val_assign(1.25, assign_error_none);
    EXPECT_EQ(1, *ptr_i8);
    a.val_assign(-129.0, assign_error_none);
    //EXPECT_EQ((int8_t)-129.0, *ptr_i8); // < this is undefined behavior
}

TEST(NDObjectAssign, ScalarAssignment_UInt16) {
    ndobject a;
    const uint16_t *ptr_u16;

    // Assignment to a uint16_t scalar
    a = ndobject(make_dtype<uint16_t>());
    ptr_u16 = (const uint16_t *)a.get_ndo()->m_data_pointer;
    a.val_assign(true);
    EXPECT_EQ(1, *ptr_u16);
    a.val_assign(false);
    EXPECT_EQ(0, *ptr_u16);
    EXPECT_THROW(a.val_assign(-1), runtime_error);
    EXPECT_THROW(a.val_assign(-1, assign_error_overflow), runtime_error);
    a.val_assign(-1, assign_error_none);
    EXPECT_EQ(65535, *ptr_u16);
    a.val_assign(1234);
    EXPECT_EQ(1234, *ptr_u16);
    a.val_assign(65535.0f);
    EXPECT_EQ(65535, *ptr_u16);
}

TEST(NDObjectAssign, ScalarAssignment_Float32) {
    ndobject a;
    const float *ptr_f32;

    // Assignment to a float scalar
    a = ndobject(make_dtype<float>());
    ptr_f32 = (const float *)a.get_ndo()->m_data_pointer;
    a.val_assign(true);
    EXPECT_EQ(1, *ptr_f32);
    a.val_assign(false);
    EXPECT_EQ(0, *ptr_f32);
    a.val_assign(-10);
    EXPECT_EQ(-10, *ptr_f32);
    a.val_assign((char)30);
    EXPECT_EQ(30, *ptr_f32);
    a.val_assign((uint16_t)58000);
    EXPECT_EQ(58000, *ptr_f32);
    a.val_assign(1.25);
    EXPECT_EQ(1.25, *ptr_f32);
    a.val_assign(1/3.0);
    EXPECT_EQ((float)(1/3.0), *ptr_f32);
    EXPECT_THROW(a.val_assign(1/3.0, assign_error_inexact), runtime_error);
    // Float32 can't represent this value exactly
    a.val_assign(33554433);
    EXPECT_EQ(33554432, *ptr_f32);
    EXPECT_THROW(a.val_assign(33554433, assign_error_inexact), runtime_error);
}

TEST(NDObjectAssign, ScalarAssignment_Float64) {
    ndobject a;
    const double *ptr_f64;

    // Assignment to a double scalar
    a = ndobject(make_dtype<double>());
    ptr_f64 = (const double *)a.get_ndo()->m_data_pointer;
    a.val_assign(true);
    EXPECT_EQ(1, *ptr_f64);
    a.val_assign(false);
    EXPECT_EQ(0, *ptr_f64);
    a.val_assign(1/3.0f);
    EXPECT_EQ(1/3.0f, *ptr_f64);
    a.val_assign(1/3.0);
    EXPECT_EQ(1/3.0, *ptr_f64);
    a.val_assign(33554433, assign_error_inexact);
    EXPECT_EQ(33554433, *ptr_f64);
    // Float64 can't represent this integer value exactly
    a.val_assign(36028797018963969LL);
    EXPECT_EQ(36028797018963968LL, *ptr_f64);
    EXPECT_THROW(a.val_assign(36028797018963969LL, assign_error_inexact), runtime_error);
}

TEST(NDObjectAssign, ScalarAssignment_Complex_Float32) {
    ndobject a;
    const complex<float> *ptr_cf32;

    // Assignment to a complex float scalar
    a = ndobject(make_dtype<complex<float> >());
    ptr_cf32 = (const complex<float> *)a.get_ndo()->m_data_pointer;
    a.val_assign(true);
    EXPECT_EQ(complex<float>(1), *ptr_cf32);
    a.val_assign(false);
    EXPECT_EQ(complex<float>(0), *ptr_cf32);
    a.val_assign(1/3.0f);
    EXPECT_EQ(complex<float>(1/3.0f), *ptr_cf32);
    a.val_assign(1/3.0);
    EXPECT_EQ(complex<float>(float(1/3.0)), *ptr_cf32);
    EXPECT_THROW(a.val_assign(1/3.0, assign_error_inexact), runtime_error);
    // Float32 can't represent this integer value exactly
    a.val_assign(33554433);
    EXPECT_EQ(33554432., ptr_cf32->real());
    EXPECT_EQ(0., ptr_cf32->imag());
    EXPECT_THROW(a.val_assign(33554433, assign_error_inexact), runtime_error);

    a.val_assign(complex<float>(1.5f, 2.75f));
    EXPECT_EQ(complex<float>(1.5f, 2.75f), *ptr_cf32);
    a.val_assign(complex<double>(1/3.0, -1/7.0));
    EXPECT_EQ(complex<float>(float(1/3.0), float(-1/7.0)), *ptr_cf32);
    EXPECT_THROW(a.val_assign(complex<double>(1/3.0, -1/7.0), assign_error_inexact), runtime_error);
}

TEST(NDObjectAssign, ScalarAssignment_Complex_Float64) {
    ndobject a;
    const complex<double> *ptr_cf64;

    // Assignment to a complex float scalar
    a = ndobject(make_dtype<complex<double> >());
    ptr_cf64 = (const complex<double> *)a.get_ndo()->m_data_pointer;
    a.val_assign(true);
    EXPECT_EQ(complex<double>(1), *ptr_cf64);
    a.val_assign(false);
    EXPECT_EQ(complex<double>(0), *ptr_cf64);
    a.val_assign(1/3.0f);
    EXPECT_EQ(complex<double>(1/3.0f), *ptr_cf64);
    a.val_assign(1/3.0);
    EXPECT_EQ(complex<double>(1/3.0), *ptr_cf64);
    a.val_assign(33554433, assign_error_inexact);
    EXPECT_EQ(33554433., ptr_cf64->real());
    EXPECT_EQ(0., ptr_cf64->imag());
    // Float64 can't represent this integer value exactly
    a.val_assign(36028797018963969LL);
    EXPECT_EQ(36028797018963968LL, ptr_cf64->real());
    EXPECT_EQ(0, ptr_cf64->imag());
    EXPECT_THROW(a.val_assign(36028797018963969LL, assign_error_inexact), runtime_error);

    a.val_assign(complex<float>(1.5f, 2.75f));
    EXPECT_EQ(complex<double>(1.5f, 2.75f), *ptr_cf64);
    a.val_assign(complex<double>(1/3.0, -1/7.0), assign_error_inexact);
    EXPECT_EQ(complex<double>(1/3.0, -1/7.0), *ptr_cf64);
}

TEST(NDObjectAssign, BroadcastAssign) {
    ndobject a = make_strided_ndobject(2, 3, 4, make_dtype<float>());
    int v0[4] = {3,4,5,6};
    ndobject b = v0;

    // Broadcasts the 4-vector by a factor of 6,
    // converting the dtype
    a.val_assign(b);
    const float *ptr_f = (const float *)a.get_readonly_originptr();
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(3, *ptr_f++);
        EXPECT_EQ(4, *ptr_f++);
        EXPECT_EQ(5, *ptr_f++);
        EXPECT_EQ(6, *ptr_f++);
    }

    float v1[4] = {1.5, 2.5, 1.25, 2.25};
    b = v1;

    // Broadcasts the 4-vector by a factor of 6,
    // doesn't convert the dtype
    a.val_assign(b);
    ptr_f = (const float *)a.get_readonly_originptr();
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(1.5, *ptr_f++);
        EXPECT_EQ(2.5, *ptr_f++);
        EXPECT_EQ(1.25, *ptr_f++);
        EXPECT_EQ(2.25, *ptr_f++);
    }

    double v2[3][1] = {{1.5}, {3.125}, {7.5}};
    b = v2;
    // Broadcasts the (3,1)-array by a factor of 8,
    // converting the dtype
    a.val_assign(b);
    ptr_f = (const float *)a.get_readonly_originptr();
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j)
            EXPECT_EQ(1.5, *ptr_f++);
        for (int j = 0; j < 4; ++j)
            EXPECT_EQ(3.125, *ptr_f++);
        for (int j = 0; j < 4; ++j)
            EXPECT_EQ(7.5, *ptr_f++);
    }

}

TEST(NDObjectAssign, Casting) {
    float v0[4] = {3.5, 1.0, 0, 1000};
    ndobject a = v0, b;

    b = a.cast_scalars(make_dtype<int>());
    // This triggers the conversion from float to int,
    // but the default assign policy is 'fractional'
    EXPECT_THROW(ndobject(b.vals()), runtime_error);

    // Allow truncation of fractional part
    b = a.cast_scalars(make_dtype<int>(), assign_error_overflow);
    b = b.vals();
    EXPECT_EQ(3, b.at(0).as<int>());
    EXPECT_EQ(1, b.at(1).as<int>());
    EXPECT_EQ(0, b.at(2).as<int>());
    EXPECT_EQ(1000, b.at(3).as<int>());

    // cast_scalars<int>() should be equivalent to cast_scalars(make_dtype<int>())
    b = a.cast_scalars<int>(assign_error_overflow);
    b = b.vals();
    EXPECT_EQ(3, b.at(0).as<int>());
    EXPECT_EQ(1, b.at(1).as<int>());
    EXPECT_EQ(0, b.at(2).as<int>());
    EXPECT_EQ(1000, b.at(3).as<int>());

    b = a.cast_scalars(make_dtype<int8_t>(), assign_error_overflow);
    // This triggers conversion from float to int8,
    // which overflows
    EXPECT_THROW(ndobject(b.vals()), runtime_error);

    // Remove the overflowing value in 'a', so b.vals() no
    // longer triggers an overflow.
    a.at(3).val_assign(-120);
    b = b.vals();
    EXPECT_EQ(3, b.at(0).as<int>());
    EXPECT_EQ(1, b.at(1).as<int>());
    EXPECT_EQ(0, b.at(2).as<int>());
    EXPECT_EQ(-120, b.at(3).as<int>());
}

TEST(NDObjectAssign, Overflow) {
    int v0[4] = {0,1,2,3};
    ndobject a = v0;

    EXPECT_THROW(a.val_assign(1e25, assign_error_overflow), runtime_error);
    EXPECT_THROW(a.val_assign(1e25f, assign_error_overflow), runtime_error);
    EXPECT_THROW(a.val_assign(-1e25, assign_error_overflow), runtime_error);
    EXPECT_THROW(a.val_assign(-1e25f, assign_error_overflow), runtime_error);
}


TEST(NDObjectAssign, ChainedCastingRead) {
    float v0[5] = {3.5f, 1.3f, -2.4999f, -2.999, 1000.50001f};
    ndobject a = v0, b;

    b = a.cast_scalars<int>(assign_error_overflow);
    b = b.cast_scalars<float>(assign_error_inexact);
    // Multiple cast_scalars operations should make a chained conversion dtype
    EXPECT_EQ(make_strided_array_dtype(
                    make_convert_dtype(make_dtype<float>(),
                                    make_convert_dtype<int, float>(assign_error_overflow), assign_error_inexact)),
              b.get_dtype());

    // Evaluating the values should truncate them to integers
    b = b.vals();
    // Now it's just the value dtype, no chaining
    EXPECT_EQ(make_strided_array_dtype(make_dtype<float>()), b.get_dtype());
    EXPECT_EQ(3, b.at(0).as<float>());
    EXPECT_EQ(1, b.at(1).as<float>());
    EXPECT_EQ(-2, b.at(2).as<float>());
    EXPECT_EQ(-2, b.at(3).as<float>());
    EXPECT_EQ(1000, b.at(4).as<float>());

    // Now try it with longer chaining through multiple element sizes
    b = a.cast_scalars<int16_t>(assign_error_overflow);
    b = b.cast_scalars<int32_t>(assign_error_overflow);
    b = b.cast_scalars<int16_t>(assign_error_overflow);
    b = b.cast_scalars<int64_t>(assign_error_overflow);
    b = b.cast_scalars<float>(assign_error_overflow);
    b = b.cast_scalars<int32_t>(assign_error_overflow);

    EXPECT_EQ(make_strided_array_dtype(
                make_convert_dtype(make_dtype<int32_t>(),
                    make_convert_dtype(make_dtype<float>(),
                        make_convert_dtype(make_dtype<int64_t>(),
                            make_convert_dtype(make_dtype<int16_t>(),
                                make_convert_dtype(make_dtype<int32_t>(),
                                    make_convert_dtype<int16_t, float>(
                                    assign_error_overflow),
                                assign_error_overflow),
                            assign_error_overflow),
                        assign_error_overflow),
                    assign_error_overflow),
                assign_error_overflow)),
            b.get_dtype());
    b = b.vals();
    EXPECT_EQ(make_strided_array_dtype(make_dtype<int32_t>()), b.get_dtype());
    EXPECT_EQ(3, b.at(0).as<int32_t>());
    EXPECT_EQ(1, b.at(1).as<int32_t>());
    EXPECT_EQ(-2, b.at(2).as<int32_t>());
    EXPECT_EQ(-2, b.at(3).as<int32_t>());
    EXPECT_EQ(1000, b.at(4).as<int32_t>());
}

TEST(NDObjectAssign, ChainedCastingWrite) {
    float v0[3] = {0, 0, 0};
    ndobject a = v0, b;

    b = a.cast_scalars<int>(assign_error_inexact);
    b = b.cast_scalars<float>(assign_error_overflow);
    // Multiple cast_scalars operations should make a chained conversion dtype
    EXPECT_EQ(make_strided_array_dtype(make_convert_dtype(make_dtype<float>(),
                                    make_convert_dtype<int, float>(assign_error_inexact), assign_error_overflow)),
              b.get_dtype());

    b.at(0).vals() = 6.8f;
    b.at(1).vals() = -3.1;
    b.at(2).vals() = 1000.5;
    // Assigning should trigger the overflow
    EXPECT_THROW(b.at(2).vals() = 1e25f, runtime_error);

    // Check that the values in a got assigned as expected
    EXPECT_EQ(6, a.at(0).as<float>());
    EXPECT_EQ(-3, a.at(1).as<float>());
    EXPECT_EQ(1000, a.at(2).as<float>());
}

TEST(NDObjectAssign, ChainedCastingReadWrite) {
    float v0[3] = {0.5, -1000, -2.2};
    int16_t v1[3] = {0, 0, 0};
    ndobject a = v0, b = v1;

    // First test with a single expression in both src and dst
    ndobject aview = a.cast_scalars<double>();
    ndobject bview = b.cast_scalars<int32_t>();

    bview.val_assign(aview, assign_error_overflow);
    EXPECT_EQ(0, b.at(0).as<int>());
    EXPECT_EQ(-1000, b.at(1).as<int>());
    EXPECT_EQ(-2, b.at(2).as<int>());

    // Now test with longer chains
    b.vals() = 123;
    aview = aview.cast_scalars<int32_t>(assign_error_overflow);
    aview = aview.cast_scalars<int16_t>(assign_error_overflow);
    bview = bview.cast_scalars<int64_t>(assign_error_overflow);

    bview.vals() = aview;
    EXPECT_EQ(0, b.at(0).as<int>());
    EXPECT_EQ(-1000, b.at(1).as<int>());
    EXPECT_EQ(-2, b.at(2).as<int>());

}
