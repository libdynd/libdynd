//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>

#include "inc_gtest.hpp"
#include "../test_memory.hpp"

#include <dynd/array.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/types/strided_dim_type.hpp>

using namespace std;
using namespace dynd;

template <typename T>
class ArrayAssign : public MemoryPair<T> {
};

TYPED_TEST_CASE_P(ArrayAssign);

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Bool) {
    nd::array a;

    // assignment to a bool scalar
    a = nd::empty(TestFixture::First::MakeType(ndt::make_type<dynd_bool>()));
    const dynd_bool *ptr_b = (const dynd_bool *)a.get_ndo()->m_data_pointer;
    a.val_assign(TestFixture::Second::To(true));
    EXPECT_TRUE(TestFixture::First::Dereference(ptr_b));
    a.val_assign(TestFixture::Second::To(false));
    EXPECT_FALSE(TestFixture::First::Dereference(ptr_b));
    a.val_assign(TestFixture::Second::To(1));
    EXPECT_TRUE(TestFixture::First::Dereference(ptr_b));
    a.val_assign(TestFixture::Second::To(0));
    EXPECT_FALSE(TestFixture::First::Dereference(ptr_b));
    a.val_assign(TestFixture::Second::To(1.0));
    EXPECT_TRUE(TestFixture::First::Dereference(ptr_b));
    a.val_assign(TestFixture::Second::To(0.0));
    EXPECT_FALSE(TestFixture::First::Dereference(ptr_b));
    a.val_assign(TestFixture::Second::To(1.5), assign_error_none);
    EXPECT_TRUE(TestFixture::First::Dereference(ptr_b));
    a.val_assign(TestFixture::Second::To(-3.5f), assign_error_none);
    EXPECT_TRUE(TestFixture::First::Dereference(ptr_b));
    a.val_assign(TestFixture::Second::To(22), assign_error_none);
    EXPECT_TRUE(TestFixture::First::Dereference(ptr_b));
    if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(2)), runtime_error);
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(-1)), runtime_error);
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(1.5)), runtime_error);
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(1.5), assign_error_overflow), runtime_error);
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(1.5), assign_error_fractional), runtime_error);
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(1.5), assign_error_inexact), runtime_error);
    }
}

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Int8) {
    nd::array a;
    const int8_t *ptr_i8;

    // Assignment to an int8_t scalar
    a = nd::empty(TestFixture::First::MakeType(ndt::make_type<int8_t>()));
    ptr_i8 = (const int8_t *)a.get_ndo()->m_data_pointer;
    a.val_assign(TestFixture::Second::To(true));
    EXPECT_EQ(1, TestFixture::First::Dereference(ptr_i8));
    a.val_assign(TestFixture::Second::To(false));
    EXPECT_EQ(0, TestFixture::First::Dereference(ptr_i8));
    a.val_assign(TestFixture::Second::To(-10));
    EXPECT_EQ(-10, TestFixture::First::Dereference(ptr_i8));
    a.val_assign(TestFixture::Second::To(-128));
    EXPECT_EQ(-128, TestFixture::First::Dereference(ptr_i8));
    a.val_assign(TestFixture::Second::To(127));
    EXPECT_EQ(127, TestFixture::First::Dereference(ptr_i8));
    if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(-129)), overflow_error);
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(128)), overflow_error);
    }
    a.val_assign(TestFixture::Second::To(5.0));
    EXPECT_EQ(5, TestFixture::First::Dereference(ptr_i8));
    a.val_assign(TestFixture::Second::To(-100.0f));
    EXPECT_EQ(-100, TestFixture::First::Dereference(ptr_i8));
    if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(1.25)), runtime_error);
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(128.0)), overflow_error);
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(128.0), assign_error_inexact), runtime_error);
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(1e30)), runtime_error);
        a.val_assign(TestFixture::Second::To(1.25), assign_error_overflow);
        EXPECT_EQ(1, TestFixture::First::Dereference(ptr_i8));
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(-129.0), assign_error_overflow), overflow_error);
    }
    a.val_assign(TestFixture::Second::To(1.25), assign_error_none);
    EXPECT_EQ(1, TestFixture::First::Dereference(ptr_i8));
    a.val_assign(TestFixture::Second::To(-129.0), assign_error_none);
    //EXPECT_EQ((int8_t)-129.0, *ptr_i8); // < this is undefined behavior*/
}

TYPED_TEST_P(ArrayAssign, ScalarAssignment_UInt16) {
    nd::array a;
    const uint16_t *ptr_u16;

    // Assignment to a uint16_t scalar
    a = nd::empty(TestFixture::First::MakeType(ndt::make_type<uint16_t>()));
    ptr_u16 = (const uint16_t *)a.get_ndo()->m_data_pointer;
    a.val_assign(TestFixture::Second::To(true));
    EXPECT_EQ(1, TestFixture::First::Dereference(ptr_u16));
    a.val_assign(TestFixture::Second::To(false));
    EXPECT_EQ(0, TestFixture::First::Dereference(ptr_u16));
    if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(-1)), overflow_error);
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(-1), assign_error_overflow), overflow_error);
    }
    a.val_assign(TestFixture::Second::To(-1), assign_error_none);
    EXPECT_EQ(65535, TestFixture::First::Dereference(ptr_u16));
    a.val_assign(TestFixture::Second::To(1234));
    EXPECT_EQ(1234, TestFixture::First::Dereference(ptr_u16));
    a.val_assign(TestFixture::Second::To(65535.0f));
    EXPECT_EQ(65535, TestFixture::First::Dereference(ptr_u16));
}

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Float32) {
    nd::array a;
    const float *ptr_f32;

    // Assignment to a float scalar
    a = nd::empty(TestFixture::First::MakeType(ndt::make_type<float>()));
    ptr_f32 = (const float *)a.get_ndo()->m_data_pointer;
    a.val_assign(TestFixture::Second::To(true));
    EXPECT_EQ(1, TestFixture::First::Dereference(ptr_f32));
    a.val_assign(TestFixture::Second::To(false));
    EXPECT_EQ(0, TestFixture::First::Dereference(ptr_f32));
    a.val_assign(TestFixture::Second::To(-10));
    EXPECT_EQ(-10, TestFixture::First::Dereference(ptr_f32));
    a.val_assign(TestFixture::Second::To((char)30));
    EXPECT_EQ(30, TestFixture::First::Dereference(ptr_f32));
    a.val_assign(TestFixture::Second::To((uint16_t)58000));
    EXPECT_EQ(58000, TestFixture::First::Dereference(ptr_f32));
    a.val_assign(TestFixture::Second::To(1.25));
    EXPECT_EQ(1.25, TestFixture::First::Dereference(ptr_f32));
    a.val_assign(TestFixture::Second::To(1/3.0));
    EXPECT_EQ((float)(1/3.0), TestFixture::First::Dereference(ptr_f32));
    if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(1/3.0), assign_error_inexact), runtime_error);
    }
    // Float32 can't represent this value exactly
    a.val_assign(TestFixture::Second::To(33554433));
    EXPECT_EQ(33554432, TestFixture::First::Dereference(ptr_f32));
    if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(33554433), assign_error_inexact), runtime_error);
    }
}

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Float64) {
    nd::array a;
    const double *ptr_f64;

    // Assignment to a double scalar
    a = nd::empty(TestFixture::First::MakeType(ndt::make_type<double>()));
    ptr_f64 = (const double *)a.get_ndo()->m_data_pointer;
    a.val_assign(TestFixture::Second::To(true));
    EXPECT_EQ(1, TestFixture::First::Dereference(ptr_f64));
    a.val_assign(TestFixture::Second::To(false));
    EXPECT_EQ(0, TestFixture::First::Dereference(ptr_f64));
    a.val_assign(TestFixture::Second::To(1/3.0f));
    EXPECT_EQ(1/3.0f, TestFixture::First::Dereference(ptr_f64));
    a.val_assign(TestFixture::Second::To(1/3.0));
    EXPECT_EQ(1/3.0, TestFixture::First::Dereference(ptr_f64));
    if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
        a.val_assign(TestFixture::Second::To(33554433), assign_error_inexact);
        EXPECT_EQ(33554433, TestFixture::First::Dereference(ptr_f64));
    }
    // Float64 can't represent this integer value exactly
    a.val_assign(TestFixture::Second::To(36028797018963969LL));
    EXPECT_EQ(36028797018963968LL, TestFixture::First::Dereference(ptr_f64));
    if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(36028797018963969LL), assign_error_inexact), runtime_error);
    }
}

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Uint64) {
    nd::array a;
    const uint64_t *ptr_u64;

    // Assignment to a double scalar
    a = nd::empty(TestFixture::First::MakeType(ndt::make_type<uint64_t>()));
    ptr_u64 = (const uint64_t *)a.get_ndo()->m_data_pointer;
    a.val_assign(TestFixture::Second::To(true));
    EXPECT_EQ(1u, TestFixture::First::Dereference(ptr_u64));
    a.val_assign(TestFixture::Second::To(false));
    EXPECT_EQ(0u, TestFixture::First::Dereference(ptr_u64));
    // Assign some values that don't fit in 32-bits
    a.val_assign(TestFixture::Second::To(1.0e10f));
    EXPECT_EQ(10000000000ULL, TestFixture::First::Dereference(ptr_u64));
    a.val_assign(TestFixture::Second::To(2.0e10));
    EXPECT_EQ(20000000000ULL, TestFixture::First::Dereference(ptr_u64));
}

#if !(defined(_WIN32) && !defined(_M_X64)) // TODO: How to mark as expected failures in googletest?

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Uint64_LargeNumbers) {
    nd::array a;
    const uint64_t *ptr_u64;

    // Assignment to a double scalar
    a = nd::empty(TestFixture::First::MakeType(ndt::make_type<uint64_t>()));
    ptr_u64 = (const uint64_t *)a.get_ndo()->m_data_pointer;
    // Assign some values that don't fit in signed 64-bits
    a.val_assign(TestFixture::Second::To(13835058055282163712.f));
    EXPECT_EQ(13835058055282163712ULL, TestFixture::First::Dereference(ptr_u64));
    a.val_assign(TestFixture::Second::To(16140901064495857664.0));
    EXPECT_EQ(16140901064495857664ULL, TestFixture::First::Dereference(ptr_u64));
    a.val_assign(TestFixture::Second::To(13835058055282163712.f), assign_error_none);
    EXPECT_EQ(13835058055282163712ULL, TestFixture::First::Dereference(ptr_u64));
    a.val_assign(TestFixture::Second::To(16140901064495857664.0), assign_error_none);
    EXPECT_EQ(16140901064495857664ULL, TestFixture::First::Dereference(ptr_u64));
}
#endif

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Complex_Float32) {
    nd::array a;
    const dynd_complex<float> *ptr_cf32;

    // Assignment to a complex float scalar
    a = nd::empty(TestFixture::First::MakeType(ndt::make_type<dynd_complex<float> >()));
    ptr_cf32 = (const dynd_complex<float> *)a.get_ndo()->m_data_pointer;
    a.val_assign(TestFixture::Second::To(true));
    EXPECT_EQ(dynd_complex<float>(1), TestFixture::First::Dereference(ptr_cf32));
    a.val_assign(TestFixture::Second::To(false));
    EXPECT_EQ(dynd_complex<float>(0), TestFixture::First::Dereference(ptr_cf32));
    a.val_assign(TestFixture::Second::To(1/3.0f));
    EXPECT_EQ(dynd_complex<float>(1/3.0f), TestFixture::First::Dereference(ptr_cf32));
    a.val_assign(TestFixture::Second::To(1/3.0));
    EXPECT_EQ(dynd_complex<float>(float(1/3.0)), TestFixture::First::Dereference(ptr_cf32));
    if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(1/3.0), assign_error_inexact), runtime_error);
    }
    // Float32 can't represent this integer value exactly
    a.val_assign(TestFixture::Second::To(33554433));
    EXPECT_EQ(33554432., TestFixture::First::Dereference(ptr_cf32).real());
    EXPECT_EQ(0., TestFixture::First::Dereference(ptr_cf32).imag());
    if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(33554433), assign_error_inexact), runtime_error);
    }

    a.val_assign(TestFixture::Second::To(dynd_complex<float>(1.5f, 2.75f)));
    EXPECT_EQ(dynd_complex<float>(1.5f, 2.75f), TestFixture::First::Dereference(ptr_cf32));
    a.val_assign(TestFixture::Second::To(dynd_complex<double>(1/3.0, -1/7.0)));
    EXPECT_EQ(dynd_complex<float>(float(1/3.0), float(-1/7.0)), TestFixture::First::Dereference(ptr_cf32));
    if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(dynd_complex<double>(1/3.0, -1/7.0)), assign_error_inexact), runtime_error);
    }
}

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Complex_Float64) {
    nd::array a;
    const dynd_complex<double> *ptr_cf64;

    // Assignment to a complex float scalar
    a = nd::empty(TestFixture::First::MakeType(ndt::make_type<dynd_complex<double> >()));
    ptr_cf64 = (const dynd_complex<double> *)a.get_ndo()->m_data_pointer;
    a.val_assign(TestFixture::Second::To(true));
    EXPECT_EQ(dynd_complex<double>(1), TestFixture::First::Dereference(ptr_cf64));
    a.val_assign(TestFixture::Second::To(false));
    EXPECT_EQ(dynd_complex<double>(0), TestFixture::First::Dereference(ptr_cf64));
    a.val_assign(TestFixture::Second::To(1/3.0f));
    EXPECT_EQ(dynd_complex<double>(1/3.0f), TestFixture::First::Dereference(ptr_cf64));
    a.val_assign(TestFixture::Second::To(1/3.0));
    EXPECT_EQ(dynd_complex<double>(1/3.0), TestFixture::First::Dereference(ptr_cf64));
    if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
        a.val_assign(TestFixture::Second::To(33554433), assign_error_inexact);
        EXPECT_EQ(33554433., TestFixture::First::Dereference(ptr_cf64).real());
        EXPECT_EQ(0., TestFixture::First::Dereference(ptr_cf64).imag());
    }
    // Float64 can't represent this integer value exactly
    a.val_assign(TestFixture::Second::To(36028797018963969LL));
    EXPECT_EQ(36028797018963968LL, TestFixture::First::Dereference(ptr_cf64).real());
    EXPECT_EQ(0, TestFixture::First::Dereference(ptr_cf64).imag());
    if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(36028797018963969LL), assign_error_inexact), runtime_error);
    }

    a.val_assign(TestFixture::Second::To(dynd_complex<float>(1.5f, 2.75f)));
    EXPECT_EQ(dynd_complex<double>(1.5f, 2.75f), TestFixture::First::Dereference(ptr_cf64));
    if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
        a.val_assign(TestFixture::Second::To(dynd_complex<double>(1/3.0, -1/7.0)), assign_error_inexact);
        EXPECT_EQ(dynd_complex<double>(1/3.0, -1/7.0), TestFixture::First::Dereference(ptr_cf64));
    }
}

TYPED_TEST_P(ArrayAssign, BroadcastAssign) {
    nd::array a = nd::make_strided_array(2, 3, 4, TestFixture::First::MakeType(ndt::make_type<float>()));
    int v0[4] = {3,4,5,6};
    nd::array b = TestFixture::Second::To(v0);

    // Broadcasts the 4-vector by a factor of 6,
    // converting the type
    a.val_assign(b);
    const float *ptr_f = (const float *)a.get_readonly_originptr();
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(3, TestFixture::First::Dereference(ptr_f++));
        EXPECT_EQ(4, TestFixture::First::Dereference(ptr_f++));
        EXPECT_EQ(5, TestFixture::First::Dereference(ptr_f++));
        EXPECT_EQ(6, TestFixture::First::Dereference(ptr_f++));
    }

    float v1[4] = {1.5, 2.5, 1.25, 2.25};
    b = TestFixture::Second::To(v1);

    // Broadcasts the 4-vector by a factor of 6,
    // doesn't convert the type
    a.val_assign(b);
    ptr_f = (const float *)a.get_readonly_originptr();
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(1.5, TestFixture::First::Dereference(ptr_f++));
        EXPECT_EQ(2.5, TestFixture::First::Dereference(ptr_f++));
        EXPECT_EQ(1.25, TestFixture::First::Dereference(ptr_f++));
        EXPECT_EQ(2.25, TestFixture::First::Dereference(ptr_f++));
    }

    double v2[3][1] = {{1.5}, {3.125}, {7.5}};
    b = TestFixture::Second::To(v2);

    // Broadcasts the (3,1)-array by a factor of 8,
    // converting the type
    a.val_assign(b);
    ptr_f = (const float *)a.get_readonly_originptr();
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j)
            EXPECT_EQ(1.5, TestFixture::First::Dereference(ptr_f++));
        for (int j = 0; j < 4; ++j)
            EXPECT_EQ(3.125, TestFixture::First::Dereference(ptr_f++));
        for (int j = 0; j < 4; ++j)
            EXPECT_EQ(7.5, TestFixture::First::Dereference(ptr_f++));
    }
}

TEST(ArrayAssign, Casting) {
    float v0[4] = {3.5, 1.0, 0, 1000};
    nd::array a = v0, b;

    b = a.ucast(ndt::make_type<int>());
    // This triggers the conversion from float to int,
    // but the default assign policy is 'fractional'
    EXPECT_THROW(b.eval(), runtime_error);

    // Allow truncation of fractional part
    b = a.ucast(ndt::make_type<int>(), 0, assign_error_overflow);
    b = b.eval();
    EXPECT_EQ(3, b(0).as<int>());
    EXPECT_EQ(1, b(1).as<int>());
    EXPECT_EQ(0, b(2).as<int>());
    EXPECT_EQ(1000, b(3).as<int>());

    // cast_scalars<int>() should be equivalent to cast_scalars(ndt::make_type<int>())
    b = a.ucast<int>(0, assign_error_overflow);
    b = b.eval();
    EXPECT_EQ(3, b(0).as<int>());
    EXPECT_EQ(1, b(1).as<int>());
    EXPECT_EQ(0, b(2).as<int>());
    EXPECT_EQ(1000, b(3).as<int>());

    b = a.ucast(ndt::make_type<int8_t>(), 0, assign_error_overflow);
    // This triggers conversion from float to int8,
    // which overflows
    EXPECT_THROW(b.eval(), runtime_error);

    // Remove the overflowing value in 'a', so b.vals() no
    // longer triggers an overflow.
    a(3).val_assign(-120);
    b = b.eval();
    EXPECT_EQ(3, b(0).as<int>());
    EXPECT_EQ(1, b(1).as<int>());
    EXPECT_EQ(0, b(2).as<int>());
    EXPECT_EQ(-120, b(3).as<int>());
}

TYPED_TEST_P(ArrayAssign, Overflow) {
    int v0[4] = {0,1,2,3};
    nd::array a = TestFixture::First::To(v0);

    if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(1e25), assign_error_overflow), runtime_error);
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(1e25f), assign_error_overflow), runtime_error);
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(-1e25), assign_error_overflow), runtime_error);
        EXPECT_THROW(a.val_assign(TestFixture::Second::To(-1e25f), assign_error_overflow), runtime_error);
    }
}


TEST(ArrayAssign, ChainedCastingRead) {
    float v0[5] = {3.5f, 1.3f, -2.4999f, -2.999f, 1000.50001f};
    nd::array a = v0, b;

    b = a.ucast<int>(0, assign_error_overflow);
    b = b.ucast<float>(0, assign_error_inexact);
    // Multiple cast_scalars operations should make a chained conversion type
    EXPECT_EQ(ndt::make_strided_dim(
                    ndt::make_convert(ndt::make_type<float>(),
                                    ndt::make_convert<int, float>(assign_error_overflow), assign_error_inexact)),
              b.get_type());

    // Evaluating the values should truncate them to integers
    b = b.eval();
    // Now it's just the value type, no chaining
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_type<float>()), b.get_type());
    EXPECT_EQ(3, b(0).as<float>());
    EXPECT_EQ(1, b(1).as<float>());
    EXPECT_EQ(-2, b(2).as<float>());
    EXPECT_EQ(-2, b(3).as<float>());
    EXPECT_EQ(1000, b(4).as<float>());

    // Now try it with longer chaining through multiple element sizes
    b = a.ucast<int16_t>(0, assign_error_overflow);
    b = b.ucast<int32_t>(0, assign_error_overflow);
    b = b.ucast<int16_t>(0, assign_error_overflow);
    b = b.ucast<int64_t>(0, assign_error_overflow);
    b = b.ucast<float>(0, assign_error_overflow);
    b = b.ucast<int32_t>(0, assign_error_overflow);

    EXPECT_EQ(ndt::make_strided_dim(
                ndt::make_convert(ndt::make_type<int32_t>(),
                    ndt::make_convert(ndt::make_type<float>(),
                        ndt::make_convert(ndt::make_type<int64_t>(),
                            ndt::make_convert(ndt::make_type<int16_t>(),
                                ndt::make_convert(ndt::make_type<int32_t>(),
                                    ndt::make_convert<int16_t, float>(
                                    assign_error_overflow),
                                assign_error_overflow),
                            assign_error_overflow),
                        assign_error_overflow),
                    assign_error_overflow),
                assign_error_overflow)),
            b.get_type());
    b = b.eval();
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_type<int32_t>()), b.get_type());
    EXPECT_EQ(3, b(0).as<int32_t>());
    EXPECT_EQ(1, b(1).as<int32_t>());
    EXPECT_EQ(-2, b(2).as<int32_t>());
    EXPECT_EQ(-2, b(3).as<int32_t>());
    EXPECT_EQ(1000, b(4).as<int32_t>());
}

TEST(ArrayAssign, ChainedCastingWrite) {
    float v0[3] = {0, 0, 0};
    nd::array a = v0, b;

    b = a.ucast<int>(0, assign_error_inexact);
    b = b.ucast<float>(0, assign_error_overflow);
    // Multiple cast_scalars operations should make a chained conversion type
    EXPECT_EQ(ndt::make_strided_dim(ndt::make_convert(ndt::make_type<float>(),
                                    ndt::make_convert<int, float>(assign_error_inexact), assign_error_overflow)),
              b.get_type());

    b(0).vals() = 6.8f;
    b(1).vals() = -3.1;
    b(2).vals() = 1000.5;
    // Assigning should trigger the overflow
    EXPECT_THROW(b(2).vals() = 1e25f, runtime_error);

    // Check that the values in a got assigned as expected
    EXPECT_EQ(6, a(0).as<float>());
    EXPECT_EQ(-3, a(1).as<float>());
    EXPECT_EQ(1000, a(2).as<float>());
}

TEST(ArrayAssign, ChainedCastingReadWrite) {
    float v0[3] = {0.5f, -1000.f, -2.2f};
    int16_t v1[3] = {0, 0, 0};
    nd::array a = v0, b = v1;

    // First test with a single expression in both src and dst
    nd::array aview = a.ucast<double>();
    nd::array bview = b.ucast<int32_t>();

    bview.val_assign(aview, assign_error_overflow);
    EXPECT_EQ(0, b(0).as<int>());
    EXPECT_EQ(-1000, b(1).as<int>());
    EXPECT_EQ(-2, b(2).as<int>());

    // Now test with longer chains
    b.vals() = 123;
    aview = aview.ucast<int32_t>(0, assign_error_overflow);
    aview = aview.ucast<int16_t>(0, assign_error_overflow);
    bview = bview.ucast<int64_t>(0, assign_error_overflow);

    bview.vals() = aview;
    EXPECT_EQ(0, b(0).as<int>());
    EXPECT_EQ(-1000, b(1).as<int>());
    EXPECT_EQ(-2, b(2).as<int>());

}

TEST(ArrayAssign, ZeroSizedAssign) {
    nd::array a = nd::empty(0, "M * float64"), b = nd::empty(0, "M * float32");
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(0, a.get_shape()[0]);
    // Should be able to assign zero-sized array to zero-sized array
    a.vals() = b;
    b.vals() = a;
    // Should be able to assign zero-sized input to a vardim output
    a = nd::empty("var * float64");
    a.vals() = b;
    EXPECT_EQ(0, a.get_dim_size());
    // With a struct
    a = nd::empty("var * {a:int32, b:string}");
    b = nd::empty(0, "M * {a:int32, b:string}");
    a.vals() = b;
}

REGISTER_TYPED_TEST_CASE_P(ArrayAssign, ScalarAssignment_Bool, ScalarAssignment_Int8, ScalarAssignment_UInt16,
    ScalarAssignment_Float32, ScalarAssignment_Float64, ScalarAssignment_Uint64, ScalarAssignment_Uint64_LargeNumbers,
    ScalarAssignment_Complex_Float32, ScalarAssignment_Complex_Float64, BroadcastAssign, Overflow);

INSTANTIATE_TYPED_TEST_CASE_P(Default, ArrayAssign, DefaultMemoryPairs);
#ifdef DYND_CUDA
INSTANTIATE_TYPED_TEST_CASE_P(CUDA, ArrayAssign, CUDAMemoryPairs);
#endif // DYND_CUDA
