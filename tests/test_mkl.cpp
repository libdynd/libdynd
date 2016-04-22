//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dynd_assertions.hpp"
#include "inc_gtest.hpp"

#include <dynd/mkl.hpp>
#include <dynd/random.hpp>

using namespace std;
using namespace dynd;

namespace {

template <typename T>
class FFT : public ::testing::Test {};

template <typename R>
using ResTypes =
    ::testing::Types<R[4], R[8], R[17], R[25], R[32], R[64], R[76], R[99], R[128], R[203], R[256], R[32][32]>;

} // unnamed namespace

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

REGISTER_TYPED_TEST_CASE_P(FFT, Linear);
INSTANTIATE_TYPED_TEST_CASE_P(MKL, FFT, ResTypes<dynd::complex<double>>);
