//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <inc_gtest.hpp>

#include "../test_memory_new.hpp"
#include "../dynd_assertions.hpp"

#include <dynd/func/arithmetic.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/array.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/func/take.hpp>
#include <dynd/func/option.hpp>

#include <dynd/types/option_type.hpp>
#include <dynd/kernels/arithmetic.hpp>

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
  EXPECT_EQ(ndt::type::make<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(1, c(0, 0).as<int>());
  EXPECT_EQ(3, c(0, 1).as<int>());
  EXPECT_EQ(4, c(0, 2).as<int>());
  EXPECT_EQ(3, c(1, 0).as<int>());
  EXPECT_EQ(7, c(1, 1).as<int>());
  EXPECT_EQ(-7, c(1, 2).as<int>());

  c = a - b;
  EXPECT_EQ(ndt::type::make<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(1, c(0, 0).as<int>());
  EXPECT_EQ(1, c(0, 1).as<int>());
  EXPECT_EQ(2, c(0, 2).as<int>());
  EXPECT_EQ(-1, c(1, 0).as<int>());
  EXPECT_EQ(-3, c(1, 1).as<int>());
  EXPECT_EQ(13, c(1, 2).as<int>());
  c = b * a;
  EXPECT_EQ(ndt::type::make<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(0, c(0, 0).as<int>());
  EXPECT_EQ(2, c(0, 1).as<int>());
  EXPECT_EQ(3, c(0, 2).as<int>());
  EXPECT_EQ(2, c(1, 0).as<int>());
  EXPECT_EQ(10, c(1, 1).as<int>());
  EXPECT_EQ(-30, c(1, 2).as<int>());
  c = b / a;
  EXPECT_EQ(ndt::type::make<int>(), c.get_dtype().without_memory_type());
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
  EXPECT_EQ(ndt::type::make<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(4, c(0).as<int>());
  EXPECT_EQ(6, c(1).as<int>());
  EXPECT_EQ(8, c(2).as<int>());
  c = b + a;
  EXPECT_EQ(ndt::type::make<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(4, c(0).as<int>());
  EXPECT_EQ(6, c(1).as<int>());
  EXPECT_EQ(8, c(2).as<int>());
  c = a - b;
  EXPECT_EQ(ndt::type::make<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(0, c(0).as<int>());
  EXPECT_EQ(2, c(1).as<int>());
  EXPECT_EQ(4, c(2).as<int>());
  c = b - a;
  EXPECT_EQ(ndt::type::make<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(0, c(0).as<int>());
  EXPECT_EQ(-2, c(1).as<int>());
  EXPECT_EQ(-4, c(2).as<int>());
  c = a * b;
  EXPECT_EQ(ndt::type::make<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(4, c(0).as<int>());
  EXPECT_EQ(8, c(1).as<int>());
  EXPECT_EQ(12, c(2).as<int>());
  c = b * a;
  EXPECT_EQ(ndt::type::make<int>(), c.get_dtype().without_memory_type());
  EXPECT_EQ(4, c(0).as<int>());
  EXPECT_EQ(8, c(1).as<int>());
  EXPECT_EQ(12, c(2).as<int>());
  c = a / b;
  EXPECT_EQ(ndt::type::make<int>(), c.get_dtype().without_memory_type());
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

TEST(Arithmetic, Minus)
{
  nd::array a = {0.0, 1.0, 2.0, 3.0, 4.0};
  EXPECT_ARRAY_EQ(nd::array({-0.0, -1.0, -2.0, -3.0, -4.0}), -a);
}

/*
TEST(Arithmetic, CompoundDiv)
{
  nd::array a{1.0, 2.0, 3.0};
  a /= {2.0, 3.0, 4.0};
}
*/


TEST(Arithmetic, OptionArithmeticInt32)
{
  nd::array NA = nd::empty(ndt::type("?int32"));
  nd::assign_na(NA);
  EXPECT_ALL_FALSE(nd::is_avail(NA + 1));
  EXPECT_ALL_FALSE(nd::is_avail(NA - 1));
  EXPECT_ALL_FALSE(nd::is_avail(NA * 1));
  EXPECT_ALL_FALSE(nd::is_avail(NA / 1));

  EXPECT_ALL_FALSE(nd::is_avail(1 + NA));
  EXPECT_ALL_FALSE(nd::is_avail(1 - NA));
  EXPECT_ALL_FALSE(nd::is_avail(1 * NA));
  EXPECT_ALL_FALSE(nd::is_avail(1 / NA));

  EXPECT_ALL_FALSE(nd::is_avail(NA + NA));
  EXPECT_ALL_FALSE(nd::is_avail(NA - NA));
  EXPECT_ALL_FALSE(nd::is_avail(NA * NA));
  EXPECT_ALL_FALSE(nd::is_avail(NA / NA));
}

TEST(Arithmetic, OptionArithmeticFloat64)
{
  nd::array NA = nd::empty(ndt::type("?float64"));
  nd::assign_na(NA);
  EXPECT_ALL_FALSE(nd::is_avail(NA + 1));
  EXPECT_ALL_FALSE(nd::is_avail(NA - 1));
  EXPECT_ALL_FALSE(nd::is_avail(NA * 1));
  EXPECT_ALL_FALSE(nd::is_avail(NA / 1));

  EXPECT_ALL_FALSE(nd::is_avail(1 + NA));
  EXPECT_ALL_FALSE(nd::is_avail(1 - NA));
  EXPECT_ALL_FALSE(nd::is_avail(1 * NA));
  EXPECT_ALL_FALSE(nd::is_avail(1 / NA));

  EXPECT_ALL_FALSE(nd::is_avail(NA + NA));
  EXPECT_ALL_FALSE(nd::is_avail(NA - NA));
  EXPECT_ALL_FALSE(nd::is_avail(NA * NA));
  EXPECT_ALL_FALSE(nd::is_avail(NA / NA));
}

TEST(Arithmetic, OptionArrayLHSInt32)
{
  nd::array data = parse_json("5 * ?int32", "[null, 0, 40, null, 1]");
  nd::array expected = nd::array{false, true, true, false, true};
  nd::array indices = {1L, 2L, 4L};

  EXPECT_ARRAY_EQ(nd::is_avail(data + 1), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data - 1), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data * 2), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data / 1), expected);

  auto add = nd::array{1, 41, 2}.ucast(ndt::type("?int32")).eval();
  auto sub = nd::array{-1, 39, 0}.ucast(ndt::type("?int32")).eval();
  auto mul = nd::array{0, 80, 2}.ucast(ndt::type("?int32")).eval();
  auto div = nd::array{0, 40, 1}.ucast(ndt::type("?int32")).eval();

  for (int i = 0; i < indices.get_dim_size(); ++i) {
      auto ind = indices(i).as<int>();
      EXPECT_EQ((data + 1)(ind).as<int>(), add(i).as<int>());
      EXPECT_EQ((data - 1)(ind).as<int>(), sub(i).as<int>());
      EXPECT_EQ((data * 2)(ind).as<int>(), mul(i).as<int>());
      EXPECT_EQ((data / 1)(ind).as<int>(), div(i).as<int>());
  }
}

TEST(Arithmetic, OptionArrayLHSFloat64)
{
  nd::array data = parse_json("5 * ?int32", "[null, 0, 40, null, 1]");
  nd::array expected = nd::array{false, true, true, false, true};
  nd::array indices = {1L, 2L, 4L};

  EXPECT_ARRAY_EQ(nd::is_avail(data + 1.0), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data - 1.0), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data * 2.0), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data / 1.0), expected);

  auto add = nd::array{1.0, 41.0, 2.0}.ucast(ndt::type("?float64")).eval();
  auto sub = nd::array{-1.0, 39.0, 0.0}.ucast(ndt::type("?float64")).eval();
  auto mul = nd::array{0.0, 80.0, 2.0}.ucast(ndt::type("?float64")).eval();
  auto div = nd::array{0.0, 40.0, 1.0}.ucast(ndt::type("?float64")).eval();

  for (int i = 0; i < indices.get_dim_size(); ++i) {
      auto ind = indices(i).as<int>();
      EXPECT_EQ((data + 1.0)(ind).as<int>(), add(i).as<int>());
      EXPECT_EQ((data - 1.0)(ind).as<int>(), sub(i).as<int>());
      EXPECT_EQ((data * 2.0)(ind).as<int>(), mul(i).as<int>());
      EXPECT_EQ((data / 1.0)(ind).as<int>(), div(i).as<int>());
  }
}

TEST(Arithmetic, OptionArrayRHS)
{
  nd::array data = parse_json("5 * ?int32", "[null, -1, 40, null, 1]");
  nd::array expected = nd::array{false, true, true, false, true};
  nd::array indices = {1L, 2L, 4L};

  EXPECT_ARRAY_EQ(nd::is_avail(1 + data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(1 - data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(1 * data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(1 / data), expected);

  auto add = nd::array{0, 41, 2}.ucast(ndt::type("?int32")).eval();
  auto sub = nd::array{2, -39, 0}.ucast(ndt::type("?int32")).eval();
  auto mul = nd::array{-2, 80, 2}.ucast(ndt::type("?int32")).eval();
  auto div = nd::array{-1, 0, 1}.ucast(ndt::type("?int32")).eval();

  for (int i = 0; i < indices.get_dim_size(); ++i) {
      auto ind = indices(i).as<int>();
      EXPECT_EQ((1 + data)(ind).as<int>(), add(i).as<int>());
      EXPECT_EQ((1 - data)(ind).as<int>(), sub(i).as<int>());
      EXPECT_EQ((2 * data)(ind).as<int>(), mul(i).as<int>());
      EXPECT_EQ((1 / data)(ind).as<int>(), div(i).as<int>());
  }
}

TEST(Arithmetic, OptionArrayOptionInt32)
{
  nd::array data = parse_json("5 * ?int32", "[null, -1, 40, null, 1]");
  nd::array expected = nd::array{false, true, true, false, true};
  nd::array indices = {1L, 2L, 4L};

  EXPECT_ARRAY_EQ(nd::is_avail(data + data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data - data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data * data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data / data), expected);

  auto add = nd::array{-2, 80, 2}.ucast(ndt::type("?int32")).eval();
  auto sub = nd::array{0, 0, 0}.ucast(ndt::type("?int32")).eval();
  auto mul = nd::array{1, 1600, 1}.ucast(ndt::type("?int32")).eval();
  auto div = nd::array{1, 1, 1}.ucast(ndt::type("?int32")).eval();

  for (int i = 0; i < indices.get_dim_size(); ++i) {
      auto ind = indices(i).as<int>();
      EXPECT_EQ((data + data)(ind).as<int>(), add(i).as<int>());
      EXPECT_EQ((data - data)(ind).as<int>(), sub(i).as<int>());
      EXPECT_EQ((data * data)(ind).as<int>(), mul(i).as<int>());
      EXPECT_EQ((data / data)(ind).as<int>(), div(i).as<int>());
  }
}

TEST(Arithmetic, OptionArrayOptionFloat64)
{
  nd::array data = parse_json("5 * ?int32", "[null, -1, 40, null, 1]");
  nd::array expected = nd::array{false, true, true, false, true};
  nd::array indices = {1L, 2L, 4L};

  EXPECT_ARRAY_EQ(nd::is_avail(data + data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data - data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data * data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data / data), expected);

  auto add = nd::array{-2, 80, 2}.ucast(ndt::type("?int32")).eval();
  auto sub = nd::array{0, 0, 0}.ucast(ndt::type("?int32")).eval();
  auto mul = nd::array{1, 1600, 1}.ucast(ndt::type("?int32")).eval();
  auto div = nd::array{1, 1, 1}.ucast(ndt::type("?int32")).eval();

  auto float_data = data.ucast(ndt::type("?float64")).eval();
  for (int i = 0; i < indices.get_dim_size(); ++i) {
      auto ind = indices(i).as<int>();
      EXPECT_EQ((data + float_data)(ind).as<double>(), add(i).as<double>());
      EXPECT_EQ((data - float_data)(ind).as<double>(), sub(i).as<double>());
      EXPECT_EQ((data * float_data)(ind).as<double>(), mul(i).as<double>());
      EXPECT_EQ((data / float_data)(ind).as<double>(), div(i).as<double>());
  }
}

TEST(Arithmetic, OptionArrayNotOptionFloat64)
{
  nd::array data = parse_json("5 * ?int32", "[null, -1, 40, null, 1]");
  nd::array not_na_data = parse_json("5 * float64", "[2, -1, 40, 30, 1]");
  nd::array expected = nd::array{false, true, true, false, true};
  nd::array indices = {1L, 2L, 4L};

  EXPECT_ARRAY_EQ(nd::is_avail(data + not_na_data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data - not_na_data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data * not_na_data), expected);
  EXPECT_ARRAY_EQ(nd::is_avail(data / not_na_data), expected);

  auto add = nd::array{-2, 80, 2}.ucast(ndt::type("?float64")).eval();
  auto sub = nd::array{0, 0, 0}.ucast(ndt::type("?float64")).eval();
  auto mul = nd::array{1, 1600, 1}.ucast(ndt::type("?float64")).eval();
  auto div = nd::array{1, 1, 1}.ucast(ndt::type("?float64")).eval();

  for (int i = 0; i < indices.get_dim_size(); ++i) {
      auto ind = indices(i).as<int>();
      EXPECT_EQ((data + data)(ind).as<double>(), add(i).as<double>());
      EXPECT_EQ((data - data)(ind).as<double>(), sub(i).as<double>());
      EXPECT_EQ((data * data)(ind).as<double>(), mul(i).as<double>());
      EXPECT_EQ((data / data)(ind).as<double>(), div(i).as<double>());
  }
}

REGISTER_TYPED_TEST_CASE_P(Arithmetic, SimpleBroadcast, StridedScalarBroadcast,
                           ScalarOnTheRight, ScalarOnTheLeft, ComplexScalar);

INSTANTIATE_TYPED_TEST_CASE_P(HostMemory, Arithmetic, HostKernelRequest);
#ifdef DYND_CUDA
INSTANTIATE_TYPED_TEST_CASE_P(CUDADeviceMemory, Arithmetic,
                              CUDADeviceKernelRequest);
#endif
