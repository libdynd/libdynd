//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>

#include "inc_gtest.hpp"
#include "../test_memory.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/types/convert_type.hpp>
#include <dynd/json_parser.hpp>

using namespace std;
using namespace dynd;

template <typename T>
class ArrayAssign : public MemoryPair<T> {
};

TYPED_TEST_CASE_P(ArrayAssign);

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Bool)
{
  nd::array a;
  eval::eval_context ectx_nocheck, ectx_overflow, ectx_fractional, ectx_inexact;
  ectx_nocheck.errmode = assign_error_nocheck;
  ectx_overflow.errmode = assign_error_overflow;
  ectx_fractional.errmode = assign_error_fractional;
  ectx_inexact.errmode = assign_error_inexact;

  // assignment to a bool scalar
  a = nd::empty(TestFixture::First::MakeType(ndt::type::make<bool1>()));
  const bool1 *ptr_a = (const bool1 *)a.get_ndo()->data.ptr;
  a.val_assign(TestFixture::Second::To(true));
  EXPECT_TRUE(TestFixture::First::Dereference(ptr_a));
  a.val_assign(TestFixture::Second::To(false));
  EXPECT_FALSE(TestFixture::First::Dereference(ptr_a));
  a.val_assign(TestFixture::Second::To(1));
  EXPECT_TRUE(TestFixture::First::Dereference(ptr_a));
  a.val_assign(TestFixture::Second::To(0));
  EXPECT_FALSE(TestFixture::First::Dereference(ptr_a));
  a.val_assign(TestFixture::Second::To(1.0));
  EXPECT_TRUE(TestFixture::First::Dereference(ptr_a));
  a.val_assign(TestFixture::Second::To(0.0));
  EXPECT_FALSE(TestFixture::First::Dereference(ptr_a));
  a.val_assign(TestFixture::Second::To(1.5), &ectx_nocheck);
  EXPECT_TRUE(TestFixture::First::Dereference(ptr_a));
  /*
    Todo: Why doesn't this test consistently pass with CUDA?
    a.val_assign(TestFixture::Second::To(-3.5f), &ectx_nocheck);
    EXPECT_TRUE(TestFixture::First::Dereference(ptr_a));
  */
  a.val_assign(TestFixture::Second::To(22), &ectx_nocheck);
  EXPECT_TRUE(TestFixture::First::Dereference(ptr_a));
  if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(2)), runtime_error);
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(-1)), runtime_error);
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(1.5)), runtime_error);
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(1.5), &ectx_overflow), runtime_error);
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(1.5), &ectx_fractional), runtime_error);
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(1.5), &ectx_inexact), runtime_error);
  }
}

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Int8)
{
  nd::array a;
  const int8_t *ptr_i8;
  eval::eval_context ectx_nocheck, ectx_overflow, ectx_inexact;
  ectx_nocheck.errmode = assign_error_nocheck;
  ectx_overflow.errmode = assign_error_overflow;
  ectx_inexact.errmode = assign_error_inexact;

  // Assignment to an int8_t scalar
  a = nd::empty(TestFixture::First::MakeType(ndt::type::make<int8_t>()));
  ptr_i8 = (const int8_t *)a.get_ndo()->data.ptr;
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
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(128.0), &ectx_inexact), runtime_error);
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(1e30)), runtime_error);
    a.val_assign(TestFixture::Second::To(1.25), &ectx_overflow);
    EXPECT_EQ(1, TestFixture::First::Dereference(ptr_i8));
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(-129.0), &ectx_overflow), overflow_error);
  }
  a.val_assign(TestFixture::Second::To(1.25), &ectx_nocheck);
  EXPECT_EQ(1, TestFixture::First::Dereference(ptr_i8));
  a.val_assign(TestFixture::Second::To(-129.0), &ectx_nocheck);
  // EXPECT_EQ((int8_t)-129.0, *ptr_i8); // < this is undefined behavior*/
}

TYPED_TEST_P(ArrayAssign, ScalarAssignment_UInt16)
{
  nd::array a;
  const uint16_t *ptr_u16;
  eval::eval_context ectx_nocheck, ectx_overflow;
  ectx_nocheck.errmode = assign_error_nocheck;
  ectx_overflow.errmode = assign_error_overflow;

  // Assignment to a uint16_t scalar
  a = nd::empty(TestFixture::First::MakeType(ndt::type::make<uint16_t>()));
  ptr_u16 = (const uint16_t *)a.get_ndo()->data.ptr;
  a.val_assign(TestFixture::Second::To(true));
  EXPECT_EQ(1, TestFixture::First::Dereference(ptr_u16));
  a.val_assign(TestFixture::Second::To(false));
  EXPECT_EQ(0, TestFixture::First::Dereference(ptr_u16));
  if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(-1)), overflow_error);
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(-1), &ectx_overflow), overflow_error);
  }
  a.val_assign(TestFixture::Second::To(-1), &ectx_nocheck);
  EXPECT_EQ(65535, TestFixture::First::Dereference(ptr_u16));
  a.val_assign(TestFixture::Second::To(1234));
  EXPECT_EQ(1234, TestFixture::First::Dereference(ptr_u16));
  a.val_assign(TestFixture::Second::To(65535.0f));
  EXPECT_EQ(65535, TestFixture::First::Dereference(ptr_u16));
}

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Float32)
{
  nd::array a;
  const float *ptr_f32;
  eval::eval_context ectx_inexact;
  ectx_inexact.errmode = assign_error_inexact;

  // Assignment to a float scalar
  a = nd::empty(TestFixture::First::MakeType(ndt::type::make<float>()));
  ptr_f32 = (const float *)a.get_ndo()->data.ptr;
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
  a.val_assign(TestFixture::Second::To(1 / 3.0));
  EXPECT_EQ((float)(1 / 3.0), TestFixture::First::Dereference(ptr_f32));
  if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(1 / 3.0), &ectx_inexact), runtime_error);
  }
  // Float32 can't represent this value exactly
  a.val_assign(TestFixture::Second::To(33554433));
  EXPECT_EQ(33554432, TestFixture::First::Dereference(ptr_f32));
  if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(33554433), &ectx_inexact), runtime_error);
  }
}

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Float64)
{
  nd::array a;
  const double *ptr_f64;
  eval::eval_context ectx_inexact;
  ectx_inexact.errmode = assign_error_inexact;

  // Assignment to a double scalar
  a = nd::empty(TestFixture::First::MakeType(ndt::type::make<double>()));
  ptr_f64 = (const double *)a.get_ndo()->data.ptr;
  a.val_assign(TestFixture::Second::To(true));
  EXPECT_EQ(1, TestFixture::First::Dereference(ptr_f64));
  a.val_assign(TestFixture::Second::To(false));
  EXPECT_EQ(0, TestFixture::First::Dereference(ptr_f64));
  a.val_assign(TestFixture::Second::To(1 / 3.0f));
  EXPECT_EQ(1 / 3.0f, TestFixture::First::Dereference(ptr_f64));
  a.val_assign(TestFixture::Second::To(1 / 3.0));
  EXPECT_EQ(1 / 3.0, TestFixture::First::Dereference(ptr_f64));
  if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
    a.val_assign(TestFixture::Second::To(33554433), &ectx_inexact);
    EXPECT_EQ(33554433, TestFixture::First::Dereference(ptr_f64));
  }
  // Float64 can't represent this integer value exactly
  a.val_assign(TestFixture::Second::To(36028797018963969LL));
  EXPECT_EQ(36028797018963968LL, TestFixture::First::Dereference(ptr_f64));
  if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(36028797018963969LL), &ectx_inexact), runtime_error);
  }
}

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Uint64)
{
  nd::array a;
  const uint64_t *ptr_u64;

  // Assignment to a double scalar
  a = nd::empty(TestFixture::First::MakeType(ndt::type::make<uint64_t>()));
  ptr_u64 = (const uint64_t *)a.get_ndo()->data.ptr;
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

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Uint64_LargeNumbers)
{
  nd::array a;
  const uint64_t *ptr_u64;
  eval::eval_context ectx_nocheck;
  ectx_nocheck.errmode = assign_error_nocheck;

  // Assignment to a double scalar
  a = nd::empty(TestFixture::First::MakeType(ndt::type::make<uint64_t>()));
  ptr_u64 = (const uint64_t *)a.get_ndo()->data.ptr;
  // Assign some values that don't fit in signed 64-bits
  a.val_assign(TestFixture::Second::To(13835058055282163712.f));
  EXPECT_EQ(13835058055282163712ULL, TestFixture::First::Dereference(ptr_u64));
  a.val_assign(TestFixture::Second::To(16140901064495857664.0));
  EXPECT_EQ(16140901064495857664ULL, TestFixture::First::Dereference(ptr_u64));
  a.val_assign(TestFixture::Second::To(13835058055282163712.f), &ectx_nocheck);
  EXPECT_EQ(13835058055282163712ULL, TestFixture::First::Dereference(ptr_u64));
  a.val_assign(TestFixture::Second::To(16140901064495857664.0), &ectx_nocheck);
  EXPECT_EQ(16140901064495857664ULL, TestFixture::First::Dereference(ptr_u64));
}
#endif

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Complex_Float32)
{
  nd::array a;
  const dynd::complex<float> *ptr_cf32;
  eval::eval_context ectx_inexact;
  ectx_inexact.errmode = assign_error_inexact;

  // Assignment to a complex float scalar
  a = nd::empty(TestFixture::First::MakeType(ndt::type::make<dynd::complex<float>>()));
  ptr_cf32 = (const dynd::complex<float> *)a.get_ndo()->data.ptr;
  a.val_assign(TestFixture::Second::To(true));
  EXPECT_EQ(dynd::complex<float>(1), TestFixture::First::Dereference(ptr_cf32));
  a.val_assign(TestFixture::Second::To(false));
  EXPECT_EQ(dynd::complex<float>(0), TestFixture::First::Dereference(ptr_cf32));
  a.val_assign(TestFixture::Second::To(1 / 3.0f));
  EXPECT_EQ(dynd::complex<float>(1 / 3.0f), TestFixture::First::Dereference(ptr_cf32));
  a.val_assign(TestFixture::Second::To(1 / 3.0));
  EXPECT_EQ(dynd::complex<float>(float(1 / 3.0)), TestFixture::First::Dereference(ptr_cf32));
  if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(1 / 3.0), &ectx_inexact), runtime_error);
  }
  // Float32 can't represent this integer value exactly
  a.val_assign(TestFixture::Second::To(33554433));
  EXPECT_EQ(33554432., TestFixture::First::Dereference(ptr_cf32).real());
  EXPECT_EQ(0., TestFixture::First::Dereference(ptr_cf32).imag());
  if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(33554433), &ectx_inexact), runtime_error);
  }

  a.val_assign(TestFixture::Second::To(dynd::complex<float>(1.5f, 2.75f)));
  EXPECT_EQ(dynd::complex<float>(1.5f, 2.75f), TestFixture::First::Dereference(ptr_cf32));
  a.val_assign(TestFixture::Second::To(dynd::complex<double>(1 / 3.0, -1 / 7.0)));
  EXPECT_EQ(dynd::complex<float>(float(1 / 3.0), float(-1 / 7.0)), TestFixture::First::Dereference(ptr_cf32));
  if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(dynd::complex<double>(1 / 3.0, -1 / 7.0)), &ectx_inexact),
                 runtime_error);
  }
}

TYPED_TEST_P(ArrayAssign, ScalarAssignment_Complex_Float64)
{
  nd::array a;
  const dynd::complex<double> *ptr_cf64;
  eval::eval_context ectx_inexact;
  ectx_inexact.errmode = assign_error_inexact;

  // Assignment to a complex float scalar
  a = nd::empty(TestFixture::First::MakeType(ndt::type::make<dynd::complex<double>>()));
  ptr_cf64 = (const dynd::complex<double> *)a.get_ndo()->data.ptr;
  a.val_assign(TestFixture::Second::To(true));
  EXPECT_EQ(dynd::complex<double>(1), TestFixture::First::Dereference(ptr_cf64));
  a.val_assign(TestFixture::Second::To(false));
  EXPECT_EQ(dynd::complex<double>(0), TestFixture::First::Dereference(ptr_cf64));
  a.val_assign(TestFixture::Second::To(1 / 3.0f));
  EXPECT_EQ(dynd::complex<double>(1 / 3.0f), TestFixture::First::Dereference(ptr_cf64));
  a.val_assign(TestFixture::Second::To(1 / 3.0));
  EXPECT_EQ(dynd::complex<double>(1 / 3.0), TestFixture::First::Dereference(ptr_cf64));
  if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
    a.val_assign(TestFixture::Second::To(33554433), &ectx_inexact);
    EXPECT_EQ(33554433., TestFixture::First::Dereference(ptr_cf64).real());
    EXPECT_EQ(0., TestFixture::First::Dereference(ptr_cf64).imag());
  }
  // Float64 can't represent this integer value exactly
  a.val_assign(TestFixture::Second::To(36028797018963969LL));
  EXPECT_EQ(36028797018963968LL, TestFixture::First::Dereference(ptr_cf64).real());
  EXPECT_EQ(0, TestFixture::First::Dereference(ptr_cf64).imag());
  if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(36028797018963969LL), &ectx_inexact), runtime_error);
  }

  a.val_assign(TestFixture::Second::To(dynd::complex<float>(1.5f, 2.75f)));
  EXPECT_EQ(dynd::complex<double>(1.5f, 2.75f), TestFixture::First::Dereference(ptr_cf64));
  if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
    a.val_assign(TestFixture::Second::To(dynd::complex<double>(1 / 3.0, -1 / 7.0)), &ectx_inexact);
    EXPECT_EQ(dynd::complex<double>(1 / 3.0, -1 / 7.0), TestFixture::First::Dereference(ptr_cf64));
  }
}

TYPED_TEST_P(ArrayAssign, BroadcastAssign)
{
  nd::array a = nd::empty(TestFixture::First::MakeType(
      ndt::make_fixed_dim(2, ndt::make_fixed_dim(3, ndt::make_fixed_dim(4, ndt::type::make<float>())))));
  int v0[4] = {3, 4, 5, 6};
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

TEST(ArrayAssign, Casting)
{
  float v0[4] = {3.5, 1.0, 0, 1000};
  nd::array a = nd::array_rw(v0), b;
  eval::eval_context tmp_ectx;

  b = a.ucast(ndt::type::make<int>());
  // This triggers the conversion from float to int,
  // but the default assign policy is 'fractional'
  EXPECT_THROW(b.eval(), runtime_error);

  // Allow truncation of fractional part
  b = a.ucast(ndt::type::make<int>());
  tmp_ectx.errmode = assign_error_overflow;
  b = b.eval(&tmp_ectx);
  EXPECT_EQ(3, b(0).as<int>());
  EXPECT_EQ(1, b(1).as<int>());
  EXPECT_EQ(0, b(2).as<int>());
  EXPECT_EQ(1000, b(3).as<int>());

  // cast_scalars<int>() should be equivalent to
  // cast_scalars(ndt::type::make<int>())
  b = a.ucast<int>(0);
  tmp_ectx.errmode = assign_error_overflow;
  b = b.eval(&tmp_ectx);
  EXPECT_EQ(3, b(0).as<int>());
  EXPECT_EQ(1, b(1).as<int>());
  EXPECT_EQ(0, b(2).as<int>());
  EXPECT_EQ(1000, b(3).as<int>());

  b = a.ucast(ndt::type::make<int8_t>());
  // This triggers conversion from float to int8,
  // which overflows
  tmp_ectx.errmode = assign_error_overflow;
  EXPECT_THROW(b.eval(), runtime_error);

  // Remove the overflowing value in 'a', so b.vals() no
  // longer triggers an overflow.
  a(3).val_assign(-120);
  tmp_ectx.errmode = assign_error_overflow;
  b = b.eval(&tmp_ectx);
  EXPECT_EQ(3, b(0).as<int>());
  EXPECT_EQ(1, b(1).as<int>());
  EXPECT_EQ(0, b(2).as<int>());
  EXPECT_EQ(-120, b(3).as<int>());
}

TYPED_TEST_P(ArrayAssign, Overflow)
{
  int v0[4] = {0, 1, 2, 3};
  nd::array a = TestFixture::First::To(v0);
  eval::eval_context ectx_overflow;
  ectx_overflow.errmode = assign_error_overflow;

  if (!TestFixture::First::IsTypeID(cuda_device_type_id) && !TestFixture::Second::IsTypeID(cuda_device_type_id)) {
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(1e25), &ectx_overflow), runtime_error);
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(1e25f), &ectx_overflow), runtime_error);
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(-1e25), &ectx_overflow), runtime_error);
    EXPECT_THROW(a.val_assign(TestFixture::Second::To(-1e25f), &ectx_overflow), runtime_error);
  }
}

TEST(ArrayAssign, ChainedCastingRead)
{
  float v0[5] = {3.5f, 1.3f, -2.4999f, -2.999f, 1000.50001f};
  nd::array a = v0, b;
  eval::eval_context tmp_ectx;

  b = a.ucast<int>();
  b = b.ucast<float>();
  // Multiple cast_scalars operations should make a chained conversion type
  EXPECT_EQ(ndt::make_fixed_dim(
                5, ndt::convert_type::make(ndt::type::make<float>(),
                                           ndt::convert_type::make(ndt::type::make<int>(), ndt::type::make<float>()))),
            b.get_type());

  // Evaluating the values should truncate them to integers
  tmp_ectx.errmode = assign_error_overflow;
  b = b.eval(&tmp_ectx);
  // Now it's just the value type, no chaining
  EXPECT_EQ(ndt::type("5 * float32"), b.get_type());
  EXPECT_EQ(3, b(0).as<float>());
  EXPECT_EQ(1, b(1).as<float>());
  EXPECT_EQ(-2, b(2).as<float>());
  EXPECT_EQ(-2, b(3).as<float>());
  EXPECT_EQ(1000, b(4).as<float>());

  // Now try it with longer chaining through multiple element sizes
  b = a.ucast<int16_t>();
  b = b.ucast<int32_t>();
  b = b.ucast<int16_t>();
  b = b.ucast<int64_t>();
  b = b.ucast<float>();
  b = b.ucast<int32_t>();

  EXPECT_EQ(ndt::make_fixed_dim(
                5, ndt::convert_type::make(
                       ndt::type::make<int32_t>(),
                       ndt::convert_type::make(
                           ndt::type::make<float>(),
                           ndt::convert_type::make(
                               ndt::type::make<int64_t>(),
                               ndt::convert_type::make(
                                   ndt::type::make<int16_t>(),
                                   ndt::convert_type::make(ndt::type::make<int32_t>(),
                                                           ndt::convert_type::make(ndt::type::make<int16_t>(),
                                                                                   ndt::type::make<float>()))))))),
            b.get_type());
  tmp_ectx.errmode = assign_error_overflow;
  b = b.eval(&tmp_ectx);
  EXPECT_EQ(ndt::type("5 * int32"), b.get_type());
  EXPECT_EQ(3, b(0).as<int32_t>());
  EXPECT_EQ(1, b(1).as<int32_t>());
  EXPECT_EQ(-2, b(2).as<int32_t>());
  EXPECT_EQ(-2, b(3).as<int32_t>());
  EXPECT_EQ(1000, b(4).as<int32_t>());
}

/**
TODO: This test has a memory leak, probably because expr_assignment_kernels.cpp
does not use the new ckernel structs. It sohuld be reenabled when that is
updated.

TEST(ArrayAssign, ChainedCastingWrite) {
    float v0[3] = {0, 0, 0};
    nd::array a = nd::array_rw(v0), b;
    eval::eval_context tmp_ectx;

    b = a.ucast<int>(0);
    b = b.ucast<float>(0);
    // Multiple cast_scalars operations should make a chained conversion type
    EXPECT_EQ(ndt::make_fixed_dim(
                  3, ndt::convert_type::make(ndt::type::make<float>(),
                                       ndt::make_convert<int, float>())),
              b.get_type());

    tmp_ectx.errmode = assign_error_overflow;
    b(0).val_assign(6.8f, &tmp_ectx);
    b(1).val_assign(-3.1, &tmp_ectx);
    b(2).val_assign(1000.5, &tmp_ectx);
    // Assigning should trigger the overflow
    EXPECT_THROW(b(2).val_assign(1e25f, &tmp_ectx), overflow_error);

    // Check that the values in a got assigned as expected
    EXPECT_EQ(6, a(0).as<float>());
    EXPECT_EQ(-3, a(1).as<float>());
    EXPECT_EQ(1000, a(2).as<float>());
}
*/

TEST(ArrayAssign, ChainedCastingReadWrite)
{
  float v0[3] = {0.5f, -1000.f, -2.2f};
  int16_t v1[3] = {0, 0, 0};
  nd::array a = nd::array_rw(v0), b = nd::array_rw(v1);
  eval::eval_context tmp_ectx;

  // First test with a single expression in both src and dst
  nd::array aview = a.ucast<double>();
  nd::array bview = b.ucast<int32_t>();

  tmp_ectx.errmode = assign_error_overflow;
  bview.val_assign(aview, &tmp_ectx);
  EXPECT_EQ(0, b(0).as<int>());
  EXPECT_EQ(-1000, b(1).as<int>());
  EXPECT_EQ(-2, b(2).as<int>());

  // Now test with longer chains
  b.vals() = 123;
  aview = aview.ucast<int32_t>(0);
  aview = aview.ucast<int16_t>(0);
  bview = bview.ucast<int64_t>(0);

  tmp_ectx.errmode = assign_error_overflow;
  bview.val_assign(aview, &tmp_ectx);
  EXPECT_EQ(0, b(0).as<int>());
  EXPECT_EQ(-1000, b(1).as<int>());
  EXPECT_EQ(-2, b(2).as<int>());
}

TEST(ArrayAssign, ZeroSizedAssign)
{
  nd::array a = nd::empty(0, "float64"), b = nd::empty(0, "float32");
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
  b = nd::empty(0, "{a:int32, b:string}");
  a.vals() = b;
}

/*
Todo: Fix this test.
TEST(ArrayAssign, VarToFixedStruct)
{
  nd::array a =
      parse_json("var * {x : string, y : int32}",
                 "[[\"Alice\", 100], [\"Bob\", 50], [\"Charlie\", 200]]");
  nd::array b = nd::empty("3 * {x : string, y : int32}");
  b.vals() = a;
  EXPECT_JSON_EQ_ARR("[[\"Alice\", 100], [\"Bob\", 50], [\"Charlie\", 200]]",
                     b);
}
*/

TEST(ArrayAssign, ArrayValsAtType)
{
  nd::array a = nd::empty(4, "int64");

  a.vals_at(irange().by(2)) = 0;
  a.vals_at(irange(1, 4, 1).by(2)) = 1;
  const int64_t *data = reinterpret_cast<const int64_t *>(a.get_readonly_originptr());
  int64_t vals[] = {0, 1, 0, 1};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(vals[i], data[i]);
  }
}

#if !(defined(_WIN32) && !defined(_M_X64)) // TODO: How to mark as expected failures in googletest?
REGISTER_TYPED_TEST_CASE_P(ArrayAssign, ScalarAssignment_Bool, ScalarAssignment_Int8, ScalarAssignment_UInt16,
                           ScalarAssignment_Float32, ScalarAssignment_Float64, ScalarAssignment_Uint64,
                           ScalarAssignment_Uint64_LargeNumbers, // This one is excluded on 32-bit
                                                                 // windows
                           ScalarAssignment_Complex_Float32, ScalarAssignment_Complex_Float64, BroadcastAssign,
                           Overflow);
#else
REGISTER_TYPED_TEST_CASE_P(ArrayAssign, ScalarAssignment_Bool, ScalarAssignment_Int8, ScalarAssignment_UInt16,
                           ScalarAssignment_Float32, ScalarAssignment_Float64, ScalarAssignment_Uint64,
                           ScalarAssignment_Complex_Float32, ScalarAssignment_Complex_Float64, BroadcastAssign,
                           Overflow);
#endif

INSTANTIATE_TYPED_TEST_CASE_P(Default, ArrayAssign, DefaultMemoryPairs);
#ifdef DYND_CUDA
INSTANTIATE_TYPED_TEST_CASE_P(CUDA, ArrayAssign, CUDAMemoryPairs);
#endif // DYND_CUDA
