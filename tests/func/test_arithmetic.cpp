//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <inc_gtest.hpp>

#include "../test_memory_new.hpp"

#include <dynd/func/arithmetic.hpp>
#include <dynd/array.hpp>
#include <dynd/json_parser.hpp>

using namespace std;
using namespace dynd;

template <typename T>
class Arithmetic : public Memory<T> {
};

TYPED_TEST_CASE_P(Arithmetic);

TYPED_TEST_P(Arithmetic, SimpleBroadcast)
{
  nd::array a, b, c;

  // Two arrays with broadcasting
  const int v0[] = {1, 2, 3};
  const int v1[][3] = {{0, 1, 1}, {2, 5, -10}};
  a = TestFixture::To(v0);
  b = TestFixture::To(v1);

  c = a + b;
  EXPECT_EQ(ndt::make_type<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(1, c(0, 0).as<int>());
  EXPECT_EQ(3, c(0, 1).as<int>());
  EXPECT_EQ(4, c(0, 2).as<int>());
  EXPECT_EQ(3, c(1, 0).as<int>());
  EXPECT_EQ(7, c(1, 1).as<int>());
  EXPECT_EQ(-7, c(1, 2).as<int>());

  c = a - b;
  EXPECT_EQ(ndt::make_type<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(1, c(0, 0).as<int>());
  EXPECT_EQ(1, c(0, 1).as<int>());
  EXPECT_EQ(2, c(0, 2).as<int>());
  EXPECT_EQ(-1, c(1, 0).as<int>());
  EXPECT_EQ(-3, c(1, 1).as<int>());
  EXPECT_EQ(13, c(1, 2).as<int>());
  c = b * a;
  EXPECT_EQ(ndt::make_type<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(0, c(0, 0).as<int>());
  EXPECT_EQ(2, c(0, 1).as<int>());
  EXPECT_EQ(3, c(0, 2).as<int>());
  EXPECT_EQ(2, c(1, 0).as<int>());
  EXPECT_EQ(10, c(1, 1).as<int>());
  EXPECT_EQ(-30, c(1, 2).as<int>());
  c = b / a;
  EXPECT_EQ(ndt::make_type<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(0, c(0, 0).as<int>());
  EXPECT_EQ(0, c(0, 1).as<int>());
  EXPECT_EQ(0, c(0, 2).as<int>());
  EXPECT_EQ(2, c(1, 0).as<int>());
  EXPECT_EQ(2, c(1, 1).as<int>());
  EXPECT_EQ(-3, c(1, 2).as<int>());
}

TYPED_TEST_P(Arithmetic, StridedScalarBroadcast)
{
  nd::array a, b, c;

  // Two arrays with broadcasting
  const int v0[] = {2, 4, 6};
  a = TestFixture::To(v0);
  b = TestFixture::To(2);

  c = a + b;
  EXPECT_EQ(ndt::make_type<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(4, c(0).as<int>());
  EXPECT_EQ(6, c(1).as<int>());
  EXPECT_EQ(8, c(2).as<int>());
  c = b + a;
  EXPECT_EQ(ndt::make_type<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(4, c(0).as<int>());
  EXPECT_EQ(6, c(1).as<int>());
  EXPECT_EQ(8, c(2).as<int>());
  c = a - b;
  EXPECT_EQ(ndt::make_type<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(0, c(0).as<int>());
  EXPECT_EQ(2, c(1).as<int>());
  EXPECT_EQ(4, c(2).as<int>());
  c = b - a;
  EXPECT_EQ(ndt::make_type<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(0, c(0).as<int>());
  EXPECT_EQ(-2, c(1).as<int>());
  EXPECT_EQ(-4, c(2).as<int>());
  c = a * b;
  EXPECT_EQ(ndt::make_type<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(4, c(0).as<int>());
  EXPECT_EQ(8, c(1).as<int>());
  EXPECT_EQ(12, c(2).as<int>());
  c = b * a;
  EXPECT_EQ(ndt::make_type<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(4, c(0).as<int>());
  EXPECT_EQ(8, c(1).as<int>());
  EXPECT_EQ(12, c(2).as<int>());
  c = a / b;
  EXPECT_EQ(ndt::make_type<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(1, c(0).as<int>());
  EXPECT_EQ(2, c(1).as<int>());
  EXPECT_EQ(3, c(2).as<int>());
}

TEST(ArithmeticOp, VarToStridedBroadcast)
{
  nd::array a, b, c;

  a = parse_json("2 * var * int32", "[[1, 2, 3], [4]]");
  b = parse_json("2 * 3 * int32", "[[5, 6, 7], [8, 9, 10]]");

  // VarDim in the first operand
  c = a + b;
  ASSERT_EQ(ndt::type("2 * 3 * int32"), c.get_type());
  ASSERT_EQ(2, c.get_shape()[0]);
  ASSERT_EQ(3, c.get_shape()[1]);
  EXPECT_EQ(6, c(0, 0).as<int>());
  EXPECT_EQ(8, c(0, 1).as<int>());
  EXPECT_EQ(10, c(0, 2).as<int>());
  EXPECT_EQ(12, c(1, 0).as<int>());
  EXPECT_EQ(13, c(1, 1).as<int>());
  EXPECT_EQ(14, c(1, 2).as<int>());

  // VarDim in the second operand
  c = b - a;
  ASSERT_EQ(ndt::type("2 * 3 * int32"), c.get_type());
  ASSERT_EQ(2, c.get_shape()[0]);
  ASSERT_EQ(3, c.get_shape()[1]);
  EXPECT_EQ(4, c(0, 0).as<int>());
  EXPECT_EQ(4, c(0, 1).as<int>());
  EXPECT_EQ(4, c(0, 2).as<int>());
  EXPECT_EQ(4, c(1, 0).as<int>());
  EXPECT_EQ(5, c(1, 1).as<int>());
  EXPECT_EQ(6, c(1, 2).as<int>());
}

TEST(ArithmeticOp, VarToVarBroadcast)
{
  nd::array a, b, c;

  a = parse_json("2 * var * int32", "[[1, 2, 3], [4]]");
  b = parse_json("2 * var * int32", "[[5], [6, 7]]");

  // VarDim in both operands, produces VarDim
  c = a + b;
  ASSERT_EQ(ndt::type("2 * var * int32"), c.get_type());
  ASSERT_EQ(2, c.get_shape()[0]);
  EXPECT_EQ(6, c(0, 0).as<int>());
  EXPECT_EQ(7, c(0, 1).as<int>());
  EXPECT_EQ(8, c(0, 2).as<int>());
  EXPECT_EQ(10, c(1, 0).as<int>());
  EXPECT_EQ(11, c(1, 1).as<int>());

  a = parse_json("2 * var * int32", "[[1, 2, 3], [4]]");
  b = parse_json("2 * 1 * int32", "[[5], [6]]");

  // VarDim in first operand, strided of size 1 in the second
  ASSERT_EQ(ndt::type("2 * var * int32"), c.get_type());
  c = a + b;
  ASSERT_EQ(2, c.get_shape()[0]);
  EXPECT_EQ(6, c(0, 0).as<int>());
  EXPECT_EQ(7, c(0, 1).as<int>());
  EXPECT_EQ(8, c(0, 2).as<int>());
  EXPECT_EQ(10, c(1, 0).as<int>());

  // Strided of size 1 in the first operand, VarDim in second
  c = b - a;
  ASSERT_EQ(ndt::type("2 * var * int32"), c.get_type());
  ASSERT_EQ(2, c.get_shape()[0]);
  EXPECT_EQ(4, c(0, 0).as<int>());
  EXPECT_EQ(3, c(0, 1).as<int>());
  EXPECT_EQ(2, c(0, 2).as<int>());
  EXPECT_EQ(2, c(1, 0).as<int>());
}

TYPED_TEST_P(Arithmetic, ScalarOnTheRight)
{
  nd::array a, b, c;

  const int v0[] = {1, 2, 3};
  a = TestFixture::To(v0);

  // A scalar on the right
  c = a + TestFixture::To(12);
  EXPECT_EQ(13, c(0).as<int>());
  EXPECT_EQ(14, c(1).as<int>());
  EXPECT_EQ(15, c(2).as<int>());
  c = a - TestFixture::To(12);
  EXPECT_EQ(-11, c(0).as<int>());
  EXPECT_EQ(-10, c(1).as<int>());
  EXPECT_EQ(-9, c(2).as<int>());
  c = a * TestFixture::To(3);
  EXPECT_EQ(3, c(0).as<int>());
  EXPECT_EQ(6, c(1).as<int>());
  EXPECT_EQ(9, c(2).as<int>());
  c = a / TestFixture::To(2);
  EXPECT_EQ(0, c(0).as<int>());
  EXPECT_EQ(1, c(1).as<int>());
  EXPECT_EQ(1, c(2).as<int>());
}

TYPED_TEST_P(Arithmetic, ScalarOnTheLeft)
{
  nd::array a, b, c;

  const int v0[] = {1, 2, 3};
  a = TestFixture::To(v0);

  // A scalar on the left
  c = TestFixture::To(-1) + a;
  EXPECT_EQ(0, c(0).as<int>());
  EXPECT_EQ(1, c(1).as<int>());
  EXPECT_EQ(2, c(2).as<int>());
  c = TestFixture::To(-1) - a;
  EXPECT_EQ(-2, c(0).as<int>());
  EXPECT_EQ(-3, c(1).as<int>());
  EXPECT_EQ(-4, c(2).as<int>());
  c = TestFixture::To(5) * a;
  EXPECT_EQ(5, c(0).as<int>());
  EXPECT_EQ(10, c(1).as<int>());
  EXPECT_EQ(15, c(2).as<int>());
  c = TestFixture::To(-6) / a;
  EXPECT_EQ(-6, c(0).as<int>());
  EXPECT_EQ(-3, c(1).as<int>());
  EXPECT_EQ(-2, c(2).as<int>());
}

TYPED_TEST_P(Arithmetic, ComplexScalar)
{
  nd::array a, c;

  // Two arrays with broadcasting
  int v0[] = {1, 2, 3};
  a = TestFixture::To(v0);

  // A complex scalar
  c = a + TestFixture::To(dynd::complex<double>(1, 2));
  EXPECT_EQ(dynd::complex<double>(2, 2), c(0).as<dynd::complex<double>>());
  EXPECT_EQ(dynd::complex<double>(3, 2), c(1).as<dynd::complex<double>>());
  EXPECT_EQ(dynd::complex<double>(4, 2), c(2).as<dynd::complex<double>>());
  c = TestFixture::To(dynd::complex<double>(0, -1)) * a;
  EXPECT_EQ(dynd::complex<double>(0, -1), c(0).as<dynd::complex<double>>());
  EXPECT_EQ(dynd::complex<double>(0, -2), c(1).as<dynd::complex<double>>());
  EXPECT_EQ(dynd::complex<double>(0, -3), c(2).as<dynd::complex<double>>());
}

TEST(Arithmetic, Plus) {
  nd::array a = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5, 9.0};
  std::cout << a << std::endl;

  std::cout << nd::plus << std::endl;
  std::cout << nd::plus(a) << std::endl;

  std::cout << nd::minus << std::endl;
  std::cout << nd::minus(a) << std::endl;

//  std::exit(-1);
}

REGISTER_TYPED_TEST_CASE_P(Arithmetic, SimpleBroadcast, StridedScalarBroadcast,
                           ScalarOnTheRight, ScalarOnTheLeft, ComplexScalar);

INSTANTIATE_TYPED_TEST_CASE_P(HostMemory, Arithmetic, HostKernelRequest);
#ifdef DYND_CUDA
INSTANTIATE_TYPED_TEST_CASE_P(CUDADeviceMemory, Arithmetic,
                              CUDADeviceKernelRequest);
#endif