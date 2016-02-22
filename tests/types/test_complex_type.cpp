//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <complex>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/array.hpp>
#include <dynd/callable_registry.hpp>
#include <dynd/math.hpp>

using namespace std;
using namespace dynd;

#define EXPECT_COMPLEX_DOUBLE_EQ(a, b)                                                                                 \
  EXPECT_DOUBLE_EQ(a.real(), b.real());                                                                                \
  EXPECT_DOUBLE_EQ(a.imag(), b.imag())

#define REL_ERROR_MAX 4E-15

template <typename T>
class ComplexType : public ::testing::Test {
};

TYPED_TEST_CASE_P(ComplexType);

TEST(Complex, Math)
{
  dynd::complex<double> z;
  typedef std::complex<double> cdbl;
  typedef std::complex<double> cdbl;

  z = dynd::complex<double>(0.0, 0.0);
  EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
  EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
  EXPECT_COMPLEX_DOUBLE_EQ(dynd::exp(z), std::exp(static_cast<std::complex<double>>(z)));
  /*
      EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 1.0), pow(cdbl(z), 1.0));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 2.0), pow(cdbl(z), 2.0));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 3.0), pow(cdbl(z), 3.0));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 7.4), pow(cdbl(z), 7.4));

      z = dynd_complex<double>(1.5, 2.0);
      EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
      EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(exp(z), exp(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 0.0), pow(cdbl(z), 0.0));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 1.0), pow(cdbl(z), 1.0));
      EXPECT_EQ_RELERR(pow(z, 2.0), dynd_complex<double>(pow(cdbl(z), 2.0)),
                       REL_ERROR_MAX);
      EXPECT_EQ_RELERR(pow(z, 3.0), dynd_complex<double>(pow(cdbl(z), 3.0)),
                       REL_ERROR_MAX);
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 7.4), pow(cdbl(z), 7.4));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(0.0, 1.0)),
  pow(cdbl(z), complex<double>(0.0, 1.0)));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(0.0, -1.0)),
  pow(cdbl(z), complex<double>(0.0, -1.0)));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(7.4, -6.3)),
  pow(cdbl(z), complex<double>(7.4, -6.3)));

      z = dynd_complex<double>(1.5, 0.0);
      EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
      EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(exp(z), exp(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 0.0), pow(cdbl(z), 0.0));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 1.0), pow(cdbl(z), 1.0));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 2.0), pow(cdbl(z), 2.0));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 3.0), pow(cdbl(z), 3.0));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 7.4), pow(cdbl(z), 7.4));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(0.0, 1.0)),
  pow(cdbl(z), complex<double>(0.0, 1.0)));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(0.0, -1.0)),
  pow(cdbl(z), complex<double>(0.0, -1.0)));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(7.4, -6.3)),
  pow(cdbl(z), complex<double>(7.4, -6.3)));

      z = dynd_complex<double>(0.0, 2.0);
      EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
      EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(exp(z), exp(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 0.0), pow(cdbl(z), 0.0));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 1.0), pow(cdbl(z), 1.0));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 2.0), pow(cdbl(z), 2.0));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, 3.0), pow(cdbl(z), 3.0));
      EXPECT_EQ_RELERR(pow(z, 7.4), dynd_complex<double>(pow(cdbl(z), 7.4)),
                       REL_ERROR_MAX);
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(0.0, 1.0)),
  pow(cdbl(z), complex<double>(0.0, 1.0)));
      EXPECT_COMPLEX_DOUBLE_EQ(pow(z, dynd_complex<double>(0.0, -1.0)),
  pow(cdbl(z), complex<double>(0.0, -1.0)));
      EXPECT_EQ_RELERR(
          pow(z, dynd_complex<double>(7.4, -6.3)),
          dynd_complex<double>(pow(cdbl(z), complex<double>(7.4, -6.3))),
          REL_ERROR_MAX);

      // Todo: pow works for both arguments complex, but there is a very small
  difference in the answers from dynd and std.
      // That's fine, but we need to specify a floating-point tolerance for
  testing.

      z = dynd_complex<double>(10.0, 0.5);
      EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
      EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(exp(z), exp(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(cos(z), cos(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(sin(z), sin(cdbl(z)));

      z = dynd_complex<double>(10.0, -0.5);
      EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
      EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(exp(z), exp(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(cos(z), cos(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(sin(z), sin(cdbl(z)));

      z = dynd_complex<double>(-10.0, 0.5);
      EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
      EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(exp(z), exp(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(cos(z), cos(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(sin(z), sin(cdbl(z)));

      z = dynd_complex<double>(-10.0, -0.5);
      EXPECT_DOUBLE_EQ(abs(z), abs(cdbl(z)));
      EXPECT_DOUBLE_EQ(arg(z), arg(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(exp(z), exp(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(log(z), log(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(sqrt(z), sqrt(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(cos(z), cos(cdbl(z)));
      EXPECT_COMPLEX_DOUBLE_EQ(sin(z), sin(cdbl(z)));
  */
}

#undef ASSERT_COMPLEX_DOUBLE_EQ

TEST(ComplexDType, Create)
{
  ndt::type d;

  // complex[float32]
  d = ndt::make_type<dynd::complex<float>>();
  EXPECT_EQ(complex_float32_id, d.get_id());
  EXPECT_EQ(complex_kind_id, d.get_base_id());
  EXPECT_EQ(8u, d.get_data_size());
  EXPECT_EQ((size_t)alignof(float), d.get_data_alignment());
  EXPECT_FALSE(d.is_expression());
  EXPECT_EQ("complex[float32]", d.str());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));

  // complex[float64]
  d = ndt::make_type<dynd::complex<double>>();
  EXPECT_EQ(complex_float64_id, d.get_id());
  EXPECT_EQ(complex_kind_id, d.get_base_id());
  EXPECT_EQ(16u, d.get_data_size());
  EXPECT_EQ((size_t)alignof(double), d.get_data_alignment());
  EXPECT_FALSE(d.is_expression());
  EXPECT_EQ("complex[float64]", d.str());
  // Roundtripping through a string
  EXPECT_EQ(d, ndt::type(d.str()));
}

TEST(ComplexType, CreateFromValue)
{
  nd::array n;

  n = dynd::complex<float>(1.5f, 2.0f);
  EXPECT_EQ(n.get_type(), ndt::make_type<dynd::complex<float>>());
  EXPECT_EQ(dynd::complex<float>(1.5f, 2.0f), n.as<dynd::complex<float>>());

  n = dynd::complex<double>(2.5, 3.0);
  EXPECT_EQ(n.get_type(), ndt::make_type<dynd::complex<double>>());
  EXPECT_EQ(dynd::complex<double>(2.5, 3.0), n.as<dynd::complex<double>>());
}

#include <dynd/func/complex.hpp>

TEST(ComplexType, Properties)
{
  nd::array n;

  n = dynd::complex<float>(1.5f, 2.0f);
  EXPECT_EQ(1.5f, n.f("real").as<float>());
  EXPECT_EQ(2.0f, n.f("imag").as<float>());

  n = dynd::complex<double>(2.5, 3.0);
  EXPECT_EQ(2.5, n.f("real").as<double>());
  EXPECT_EQ(3.0, n.f("imag").as<double>());

  dynd::complex<double> avals[3] = {dynd::complex<double>(1, 2), dynd::complex<double>(-1, 1.5),
                                    dynd::complex<double>(3, 21.75)};
  n = avals;
  EXPECT_EQ(1., n.f("real")(0).as<double>());
  EXPECT_EQ(2., n.f("imag")(0).as<double>());
  EXPECT_EQ(-1., n.f("real")(1).as<double>());
  EXPECT_EQ(1.5, n.f("imag")(1).as<double>());
  EXPECT_EQ(3., n.f("real")(2).as<double>());
  EXPECT_EQ(21.75, n.f("imag")(2).as<double>());
}

TYPED_TEST_P(ComplexType, Arithmetic)
{
  EXPECT_EQ(std::complex<TypeParam>(1.5, 0.5) + static_cast<TypeParam>(1), dynd::complex<TypeParam>(1.5, 0.5) + 1);
  EXPECT_EQ(static_cast<TypeParam>(1) + std::complex<TypeParam>(1.5, 0.5), 1 + dynd::complex<TypeParam>(1.5, 0.5));

  EXPECT_EQ(std::complex<TypeParam>(1.3, 0.7) - 1.0, dynd::complex<TypeParam>(1.3, 0.7) - 1.0);
  EXPECT_EQ(1.0 - std::complex<TypeParam>(1.3, 0.7), 1.0 - dynd::complex<TypeParam>(1.3, 0.7));

  EXPECT_EQ(std::complex<TypeParam>(1.5, 0.5) * 2.0, dynd::complex<TypeParam>(1.5, 0.5) * 2.0);
  EXPECT_EQ(2.0 * std::complex<TypeParam>(1.5, 0.5), 2.0 * dynd::complex<TypeParam>(1.5, 0.5));

  EXPECT_EQ(std::complex<TypeParam>(1.5, 0.5) / static_cast<TypeParam>(2), dynd::complex<TypeParam>(1.5, 0.5) / 2.0f);
  EXPECT_COMPLEX_DOUBLE_EQ(static_cast<TypeParam>(2) / std::complex<TypeParam>(1.5, 0.5),
                           2.0f / dynd::complex<TypeParam>(1.5, 0.5));
}

REGISTER_TYPED_TEST_CASE_P(ComplexType, Arithmetic);

INSTANTIATE_TYPED_TEST_CASE_P(Double, ComplexType, double);
