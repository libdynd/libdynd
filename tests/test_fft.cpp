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

TEST(MKL, FFT) {
  nd::array x0 = nd::random::uniform({}, {{"dst_tp", ndt::make_type<dynd::complex<double>[32]>()}});
  nd::array x1 = nd::random::uniform({}, {{"dst_tp", ndt::make_type<dynd::complex<double>[32]>()}});
  nd::array x = x0 + x1;

  nd::array y0 = nd::mkl::fft(x0);
  nd::array y1 = nd::mkl::fft(x1);
  nd::array y = nd::mkl::fft(x);

  EXPECT_ARRAY_NEAR(y0 + y1, y);
}
