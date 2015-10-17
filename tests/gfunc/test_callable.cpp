//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/gfunc/gcallable.hpp>
#include <dynd/gfunc/make_gcallable.hpp>
#include <dynd/gfunc/call_gcallable.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/struct_type.hpp>

using namespace std;
using namespace dynd;

static int one_parameter(int x)
{
  return 3 * x;
}

TEST(GFuncCallable, OneParameter)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable(&one_parameter, "x");
  EXPECT_EQ(ndt::struct_type::make({"x"}, {ndt::type::make<int>()}), c.get_parameters_type());

  // Call it with the generic interface and see that it gave what we want
  nd::array a, r;
  a = nd::empty(c.get_parameters_type());

  a(0).val_assign(12);
  r = c.call_generic(a);
  EXPECT_EQ(ndt::type::make<int>(), r.get_type());
  EXPECT_EQ(36, r.as<int>());

  a(0).val_assign(3);
  r = c.call_generic(a);
  EXPECT_EQ(ndt::type::make<int>(), r.get_type());
  EXPECT_EQ(9, r.as<int>());

  // Also call it through the C++ interface
  EXPECT_EQ(3, c.call(1).as<int>());
  EXPECT_EQ(-15, c.call(-5).as<int>());
  // Should throw with the wrong number of arguments
  EXPECT_THROW(c.call(), runtime_error);
  EXPECT_THROW(c.call(1, 2), runtime_error);
}

TEST(GFuncCallable, OneParameterWithDefault)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable_with_default(&one_parameter, "x", 12);
  EXPECT_EQ(ndt::struct_type::make({"x"}, {ndt::type::make<int>()}), c.get_parameters_type());

  // Call it through the C++ interface with and without a parameter
  EXPECT_EQ(3, c.call(1).as<int>());
  EXPECT_EQ(-15, c.call(-5).as<int>());
  EXPECT_EQ(36, c.call().as<int>());
  // Should throw with the wrong number of arguments
  EXPECT_THROW(c.call(1, 2), runtime_error);
}

static double two_parameters(double a, long b)
{
  return a * b;
}

TEST(GFuncCallable, TwoParameters)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable(&two_parameters, "a", "b");
  EXPECT_EQ(ndt::struct_type::make({"a", "b"}, {ndt::type::make<double>(), ndt::type::make<long>()}),
            c.get_parameters_type());

  // Call it and see that it gave what we want
  nd::array a, r;
  a = nd::empty(c.get_parameters_type());

  a(0).val_assign(2.25);
  a(1).val_assign(3);
  r = c.call_generic(a);
  EXPECT_EQ(ndt::type::make<double>(), r.get_type());
  EXPECT_EQ(6.75, r.as<double>());

  a(0).val_assign(-1.5);
  a(1).val_assign(2);
  r = c.call_generic(a);
  EXPECT_EQ(ndt::type::make<double>(), r.get_type());
  EXPECT_EQ(-3, r.as<double>());
}

TEST(GFuncCallable, TwoParametersWithOneDefault)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable_with_default(&two_parameters, "a", "b", 5);
  EXPECT_EQ(ndt::struct_type::make({"a", "b"}, {ndt::type::make<double>(), ndt::type::make<long>()}),
            c.get_parameters_type());

  // Call it through the C++ interface with various numbers of parameters
  EXPECT_EQ(15, c.call(3, 5).as<double>());
  EXPECT_EQ(-4.5, c.call(2.25, -2).as<double>());
  EXPECT_EQ(-7.5, c.call(-1.5).as<double>());
  EXPECT_EQ(-10, c.call(-2).as<double>());
  // Should throw with the wrong number of arguments
  EXPECT_THROW(c.call(), runtime_error);
}

TEST(GFuncCallable, TwoParametersWithTwoDefaults)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable_with_default(&two_parameters, "a", "b", 1.5, 7);
  EXPECT_EQ(ndt::struct_type::make({"a", "b"}, {ndt::type::make<double>(), ndt::type::make<long>()}),
            c.get_parameters_type());

  // Call it through the C++ interface with and without a parameter
  EXPECT_EQ(15, c.call(3, 5).as<double>());
  EXPECT_EQ(-4.5, c.call(2.25, -2).as<double>());
  EXPECT_EQ(-10.5, c.call(-1.5).as<double>());
  EXPECT_EQ(-14, c.call(-2).as<double>());
  EXPECT_EQ(10.5, c.call().as<double>());
}

static dynd::complex<float> three_parameters(bool x, int a, int b)
{
  if (x) {
    return dynd::complex<float>((float)a, (float)b);
  } else {
    return dynd::complex<float>((float)b, (float)a);
  }
}

TEST(GFuncCallable, ThreeParameters)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable(&three_parameters, "s", "a", "b");
  EXPECT_EQ(ndt::struct_type::make({"s", "a", "b"},
                                   {ndt::type::make<bool1>(), ndt::type::make<int>(), ndt::type::make<int>()}),
            c.get_parameters_type());

  // Call it and see that it gave what we want
  nd::array a, r;
  a = nd::empty(c.get_parameters_type());

  a(0).val_assign(true);
  a(1).val_assign(3);
  a(2).val_assign(4);
  r = c.call_generic(a);
  EXPECT_EQ(ndt::type::make<dynd::complex<float>>(), r.get_type());
  EXPECT_EQ(dynd::complex<float>(3, 4), r.as<dynd::complex<float>>());

  a(0).val_assign(false);
  a(1).val_assign(5);
  a(2).val_assign(6);
  r = c.call_generic(a);
  EXPECT_EQ(ndt::type::make<dynd::complex<float>>(), r.get_type());
  EXPECT_EQ(dynd::complex<float>(6, 5), r.as<dynd::complex<float>>());
}

TEST(GFuncCallable, ThreeParametersWithOneDefault)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable_with_default(&three_parameters, "s", "a", "b", 12);
  EXPECT_EQ(ndt::struct_type::make({"s", "a", "b"},
                                   {ndt::type::make<bool1>(), ndt::type::make<int>(), ndt::type::make<int>()}),
            c.get_parameters_type());

  // Call it through the C++ interface with various numbers of parameters
  EXPECT_EQ(dynd::complex<float>(3, 4), c.call(true, 3, 4).as<dynd::complex<float>>());
  EXPECT_EQ(dynd::complex<float>(6, 5), c.call(false, 5, 6).as<dynd::complex<float>>());
  EXPECT_EQ(dynd::complex<float>(7, 12), c.call(true, 7).as<dynd::complex<float>>());
  EXPECT_EQ(dynd::complex<float>(12, 5), c.call(false, 5).as<dynd::complex<float>>());
  // Should throw with the wrong number of arguments
  EXPECT_THROW(c.call(), runtime_error);
  EXPECT_THROW(c.call(false), runtime_error);
  EXPECT_THROW(c.call(false, 1.5, 2, 12), runtime_error);
}

TEST(GFuncCallable, ThreeParametersWithTwoDefaults)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable_with_default(&three_parameters, "s", "a", "b", 6, 12);
  EXPECT_EQ(ndt::struct_type::make({"s", "a", "b"},
                                   {ndt::type::make<bool1>(), ndt::type::make<int>(), ndt::type::make<int>()}),
            c.get_parameters_type());

  // Call it through the C++ interface with various numbers of parameters
  EXPECT_EQ(dynd::complex<float>(3, 4), c.call(true, 3, 4).as<dynd::complex<float>>());
  EXPECT_EQ(dynd::complex<float>(6, 5), c.call(false, 5, 6).as<dynd::complex<float>>());
  EXPECT_EQ(dynd::complex<float>(7, 12), c.call(true, 7).as<dynd::complex<float>>());
  EXPECT_EQ(dynd::complex<float>(12, 5), c.call(false, 5).as<dynd::complex<float>>());
  EXPECT_EQ(dynd::complex<float>(6, 12), c.call(true).as<dynd::complex<float>>());
  EXPECT_EQ(dynd::complex<float>(12, 6), c.call(false).as<dynd::complex<float>>());
  // Should throw with the wrong number of arguments
  EXPECT_THROW(c.call(), runtime_error);
  EXPECT_THROW(c.call(false, 1.5, 2, 12), runtime_error);
}

TEST(GFuncCallable, ThreeParametersWithThreeDefaults)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable_with_default(&three_parameters, "s", "a", "b", false, 6, 12);
  EXPECT_EQ(ndt::struct_type::make({"s", "a", "b"},
                                   {ndt::type::make<bool1>(), ndt::type::make<int>(), ndt::type::make<int>()}),
            c.get_parameters_type());

  // Call it through the C++ interface with various numbers of parameters
  EXPECT_EQ(dynd::complex<float>(3, 4), c.call(true, 3, 4).as<dynd::complex<float>>());
  EXPECT_EQ(dynd::complex<float>(6, 5), c.call(false, 5, 6).as<dynd::complex<float>>());
  EXPECT_EQ(dynd::complex<float>(7, 12), c.call(true, 7).as<dynd::complex<float>>());
  EXPECT_EQ(dynd::complex<float>(12, 5), c.call(false, 5).as<dynd::complex<float>>());
  EXPECT_EQ(dynd::complex<float>(6, 12), c.call(true).as<dynd::complex<float>>());
  EXPECT_EQ(dynd::complex<float>(12, 6), c.call(false).as<dynd::complex<float>>());
  EXPECT_EQ(dynd::complex<float>(12, 6), c.call().as<dynd::complex<float>>());
  // Should throw with the wrong number of arguments
  EXPECT_THROW(c.call(false, 1.5, 2, 12), runtime_error);
}

static uint8_t four_parameters(int8_t x, int16_t y, double alpha, uint32_t z)
{
  return (uint8_t)(x * (1 - alpha) + y * alpha + z);
}

TEST(GFuncCallable, FourParameters)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable(&four_parameters, "x", "y", "alpha", "z");
  EXPECT_EQ(ndt::struct_type::make({"x", "y", "alpha", "z"}, {ndt::type::make<int8_t>(), ndt::type::make<int16_t>(),
                                                              ndt::type::make<double>(), ndt::type::make<uint32_t>()}),
            c.get_parameters_type());

  // Call it and see that it gave what we want
  nd::array a, r;
  a = nd::empty(c.get_parameters_type());

  a(0).val_assign(-1);
  a(1).val_assign(7);
  a(2).val_assign(0.25);
  a(3).val_assign(3);
  r = c.call_generic(a);
  EXPECT_EQ(ndt::type::make<uint8_t>(), r.get_type());
  EXPECT_EQ(4, r.as<uint8_t>());

  a(0).val_assign(1);
  a(1).val_assign(3);
  a(2).val_assign(0.5);
  a(3).val_assign(12);
  r = c.call_generic(a);
  EXPECT_EQ(ndt::type::make<uint8_t>(), r.get_type());
  EXPECT_EQ(14, r.as<uint8_t>());
}

TEST(GFuncCallable, FourParametersWithOneDefault)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable_with_default(&four_parameters, "x", "y", "alpha", "z", 240u);
  EXPECT_EQ(ndt::struct_type::make({"x", "y", "alpha", "z"}, {ndt::type::make<int8_t>(), ndt::type::make<int16_t>(),
                                                              ndt::type::make<double>(), ndt::type::make<uint32_t>()}),
            c.get_parameters_type());

  // Call it through the C++ interface with various numbers of parameters
  EXPECT_EQ(4u, c.call(-1, 7, 0.25, 3).as<uint8_t>());
  EXPECT_EQ(14u, c.call(1, 3, 0.5, 12).as<uint8_t>());
  EXPECT_EQ(242u, c.call(1, 3, 0.5).as<uint8_t>());
  // Should throw with the wrong number of arguments
  EXPECT_THROW(c.call(), runtime_error);
  EXPECT_THROW(c.call(2), runtime_error);
  EXPECT_THROW(c.call(2, 5), runtime_error);
  EXPECT_THROW(c.call(2, 5, 0.1, 3, 9), runtime_error);
}

TEST(GFuncCallable, FourParametersWithTwoDefaults)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable_with_default(&four_parameters, "x", "y", "alpha", "z", 0.75, 240u);
  EXPECT_EQ(ndt::struct_type::make({"x", "y", "alpha", "z"}, {ndt::type::make<int8_t>(), ndt::type::make<int16_t>(),
                                                              ndt::type::make<double>(), ndt::type::make<uint32_t>()}),
            c.get_parameters_type());

  // Call it through the C++ interface with various numbers of parameters
  EXPECT_EQ(4u, c.call(-1, 7, 0.25, 3).as<uint8_t>());
  EXPECT_EQ(14u, c.call(1, 3, 0.5, 12).as<uint8_t>());
  EXPECT_EQ(242u, c.call(1, 3, 0.5).as<uint8_t>());
  EXPECT_EQ(245u, c.call(-1, 7).as<uint8_t>());
  // Should throw with the wrong number of arguments
  EXPECT_THROW(c.call(), runtime_error);
  EXPECT_THROW(c.call(2), runtime_error);
  EXPECT_THROW(c.call(2, 5, 0.1, 3, 9), runtime_error);
}

TEST(GFuncCallable, FourParametersWithThreeDefaults)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable_with_default(&four_parameters, "x", "y", "alpha", "z", 8, 0.75, 240u);
  EXPECT_EQ(ndt::struct_type::make({"x", "y", "alpha", "z"}, {ndt::type::make<int8_t>(), ndt::type::make<int16_t>(),
                                                              ndt::type::make<double>(), ndt::type::make<uint32_t>()}),
            c.get_parameters_type());

  // Call it through the C++ interface with various numbers of parameters
  EXPECT_EQ(4u, c.call(-1, 7, 0.25, 3).as<uint8_t>());
  EXPECT_EQ(14u, c.call(1, 3, 0.5, 12).as<uint8_t>());
  EXPECT_EQ(242u, c.call(1, 3, 0.5).as<uint8_t>());
  EXPECT_EQ(245u, c.call(-1, 7).as<uint8_t>());
  EXPECT_EQ(246u, c.call(0).as<uint8_t>());
  // Should throw with the wrong number of arguments
  EXPECT_THROW(c.call(), runtime_error);
  EXPECT_THROW(c.call(2, 5, 0.1, 3, 9), runtime_error);
}

TEST(GFuncCallable, FourParametersWithFourDefaults)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable_with_default(&four_parameters, "x", "y", "alpha", "z", -8, 8, 0.75, 240u);
  EXPECT_EQ(ndt::struct_type::make({"x", "y", "alpha", "z"}, {ndt::type::make<int8_t>(), ndt::type::make<int16_t>(),
                                                              ndt::type::make<double>(), ndt::type::make<uint32_t>()}),
            c.get_parameters_type());

  // Call it through the C++ interface with various numbers of parameters
  EXPECT_EQ(4u, c.call(-1, 7, 0.25, 3).as<uint8_t>());
  EXPECT_EQ(14u, c.call(1, 3, 0.5, 12).as<uint8_t>());
  EXPECT_EQ(242u, c.call(1, 3, 0.5).as<uint8_t>());
  EXPECT_EQ(245u, c.call(-1, 7).as<uint8_t>());
  EXPECT_EQ(246u, c.call(0).as<uint8_t>());
  EXPECT_EQ(244u, c.call().as<uint8_t>());
  // Should throw with the wrong number of arguments
  EXPECT_THROW(c.call(2, 5, 0.1, 3, 9), runtime_error);
}

static double five_parameters(float (&x)[3], uint16_t a1, uint32_t a2, uint64_t a3, double (&y)[3])
{
  return x[0] * a1 * y[0] + x[1] * a2 * y[1] + x[2] * a3 * y[2];
}

TEST(GFuncCallable, FiveParameters)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable(&five_parameters, "x", "a1", "a2", "a3", "y");
  EXPECT_EQ(ndt::struct_type::make({"x", "a1", "a2", "a3", "y"},
                                   {ndt::make_fixed_dim(3, ndt::type::make<float>()), ndt::type::make<uint16_t>(),
                                    ndt::type::make<uint32_t>(),                      ndt::type::make<uint64_t>(),
                                    ndt::make_fixed_dim(3, ndt::type::make<double>())}),
            c.get_parameters_type());

  // Call it and see that it gave what we want
  nd::array a, r;
  a = nd::empty(c.get_parameters_type());

  float f0[3] = {1, 2, 3};
  double d0[3] = {1.5, 2.5, 3.5};
  a(0).val_assign(f0);
  a(1).val_assign(2);
  a(2).val_assign(4);
  a(3).val_assign(6);
  a(4).val_assign(d0);
  r = c.call_generic(a);
  EXPECT_EQ(ndt::type::make<double>(), r.get_type());
  EXPECT_EQ(86, r.as<double>());
}

static nd::array array_return(int a, int b, int c)
{
  nd::array result = nd::empty<int[3]>();
  result(0).vals() = a;
  result(1).vals() = b;
  result(2).vals() = c;
  return result;
}

TEST(GFuncCallable, ArrayReturn)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable(&array_return, "a", "b", "c");

  // Call it and see that it gave what we want
  nd::array a, r;
  a = nd::empty(c.get_parameters_type());

  a(0).val_assign(-10);
  a(1).val_assign(20);
  a(2).val_assign(1000);
  r = c.call_generic(a);
  EXPECT_EQ(ndt::make_fixed_dim(3, ndt::type::make<int>()), r.get_type());
  EXPECT_EQ(-10, r(0).as<int>());
  EXPECT_EQ(20, r(1).as<int>());
  EXPECT_EQ(1000, r(2).as<int>());
}

static size_t array_param(const nd::array &n)
{
  return n.get_type().get_ndim();
}

TEST(GFuncCallable, ArrayParam)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable(&array_param, "n");

  // Call it and see that it gave what we want
  nd::array tmp;
  nd::array a, r;
  a = nd::empty(c.get_parameters_type());

  tmp = nd::empty<int[2][3][1]>();
  *(void **)a.get_ndo()->data.ptr = tmp.get_ndo();
  r = c.call_generic(a);
  EXPECT_EQ(ndt::type::make<size_t>(), r.get_type());
  EXPECT_EQ(3, r.as<int>());
}

static size_t ndt_type_param(const ndt::type &d)
{
  return d.get_default_data_size();
}

TEST(GFuncCallable, DTypeParam)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable(&ndt_type_param, "d");

  // Call it and see that it gave what we want
  ndt::type tmp;
  nd::array a, r;
  a = nd::empty(c.get_parameters_type());

  // With a base_type
  tmp = ndt::struct_type::make({"A", "B"}, {ndt::type::make<dynd::complex<float>>(), ndt::type::make<int8_t>()});
  *(const void **)a.get_ndo()->data.ptr = tmp.extended();
  r = c.call_generic(a);
  EXPECT_EQ(ndt::type::make<size_t>(), r.get_type());
  EXPECT_EQ(12u, r.as<size_t>());

  // With a builtin type
  tmp = ndt::type::make<uint64_t>();
  *(void **)a.get_ndo()->data.ptr = (void *)tmp.get_type_id();
  r = c.call_generic(a);
  EXPECT_EQ(ndt::type::make<size_t>(), r.get_type());
  EXPECT_EQ(8u, r.as<size_t>());
}

static std::string string_return(int a, int b, int c)
{
  stringstream ss;
  ss << a << ", " << b << ", " << c;
  return ss.str();
}

TEST(GFuncCallable, StringReturn)
{
  // Create the callable
  gfunc::callable c = gfunc::make_callable(&string_return, "a", "b", "c");

  // Call it and see that it gave what we want
  nd::array a, r;
  a = nd::empty(c.get_parameters_type());

  a(0).val_assign(-10);
  a(1).val_assign(20);
  a(2).val_assign(1000);
  r = c.call_generic(a);
  EXPECT_EQ(ndt::string_type::make(), r.get_type());
  EXPECT_EQ("-10, 20, 1000", r.as<std::string>());
}
