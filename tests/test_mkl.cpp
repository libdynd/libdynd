//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <dynd/mkl.hpp>
#include <dynd/random.hpp>
#include <dynd/registry.hpp>
#include <dynd_assertions.hpp>

using namespace std;
using namespace dynd;

namespace {

template <typename T>
class FFT : public ::testing::Test {};

template <typename T>
class Conv : public ::testing::Test {};

template <typename T>
using make_types =
    ::testing::Types<T[4], T[8], T[17], T[25], T[32], T[64], T[76], T[99], T[128], T[203], T[256], T[32][32]>;

} // unnamed namespace

TEST(MKL, Load) {
  load("libdynd_mkl");

  registry_entry &entry = registered("dynd.nd.mkl");
  EXPECT_EQ(nd::mkl::fft, entry["fft"].value());
  EXPECT_EQ(nd::mkl::ifft, entry["ifft"].value());
  EXPECT_EQ(nd::mkl::conv, entry["conv"].value());
}

TYPED_TEST_CASE_P(FFT);

TYPED_TEST_P(FFT, Linear) {
  const ndt::type &res_tp = ndt::make_type<TypeParam>();

  nd::array x0 = nd::random::uniform({}, {{"dst_tp", res_tp}});
  nd::array x1 = nd::random::uniform({}, {{"dst_tp", res_tp}});
  nd::array x = x0 + x1;

  nd::array y0 = nd::mkl::fft(x0);
  nd::array y1 = nd::mkl::fft(x1);
  nd::array y = nd::mkl::fft(x);

  EXPECT_ARRAY_NEAR(y0 + y1, y);
}

/*
TYPED_TEST_P(FFT, Inverse) {
  typedef double real_type;

  const ndt::type &res_tp = ndt::make_type<TypeParam>();

  nd::array x = nd::random::uniform({}, {{"dst_tp", res_tp}});

  nd::array y = nd::mkl::ifft(nd::mkl::fft(x));
  EXPECT_ARRAY_NEAR(x, y / y.get_dim_size());

  real_type scale = 1.0 / y.get_dim_size();
  y = nd::mkl::ifft({nd::mkl::fft(x)}, {{"scale", scale}});
  EXPECT_ARRAY_NEAR(x, y);
}

TYPED_TEST_P(FFT, Zeros) {
  const ndt::type &res_tp = ndt::make_type<TypeParam>();

  nd::array x = nd::empty(res_tp).assign(dynd::complex<double>(0.0, 0.0));
  nd::array y = nd::mkl::fft(x);

  EXPECT_ARRAY_NEAR(dynd::complex<double>(0.0), y);
}
*/

TYPED_TEST_CASE_P(Conv);

TYPED_TEST_P(Conv, Linear) {
  const ndt::type &arg0_tp = ndt::make_type<TypeParam>();
  const ndt::type &arg1_tp = ndt::make_type<TypeParam>();

  nd::array x0 = nd::random::uniform({}, {{"dst_tp", arg0_tp}});
  nd::array x1 = nd::random::uniform({}, {{"dst_tp", arg0_tp}});
  nd::array x = x0 + x1;

  nd::array h = nd::random::uniform({}, {{"dst_tp", arg1_tp}});

  nd::array y0 = nd::mkl::conv(x0, h);
  nd::array y1 = nd::mkl::conv(x1, h);
  nd::array y = nd::mkl::conv(x, h);

  EXPECT_ARRAY_NEAR(y0 + y1, y);
}

/*
TYPED_TEST_P(Conv, Zeros) {
  typedef typename std::remove_extent<TypeParam>::type data_type;

  const ndt::type &arg0_tp = ndt::make_type<TypeParam>();
  const ndt::type &arg1_tp = ndt::make_type<TypeParam>();

  nd::array x = nd::empty(arg0_tp).assign(0.0);

  nd::array h = nd::random::uniform({}, {{"dst_tp", arg1_tp}});

  nd::array y = nd::mkl::conv(x, h);

  EXPECT_ARRAY_NEAR(0.0, y);
}
*/

REGISTER_TYPED_TEST_CASE_P(FFT, Linear);
INSTANTIATE_TYPED_TEST_CASE_P(MKL, FFT, make_types<dynd::complex<double>>);

REGISTER_TYPED_TEST_CASE_P(Conv, Linear);
INSTANTIATE_TYPED_TEST_CASE_P(Float64, Conv, make_types<double>);
INSTANTIATE_TYPED_TEST_CASE_P(Complex64, Conv, make_types<dynd::complex<double>>);
