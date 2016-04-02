//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"
#include "dynd_assertions.hpp"

#include <dynd/fft.hpp>
#include <dynd/random.hpp>

using namespace std;
using namespace dynd;

#ifdef DYND_FFTW

class FFT1D : public ::testing::TestWithParam<std::tr1::tuple<const char *, const char *, const char *>> {

public:
  ndt::type GetParam() const
  {
    ndt::type concrete_tp = ndt::type(std::tr1::get<1>(
        ::testing::TestWithParam<std::tr1::tuple<const char *, const char *, const char *>>::GetParam()));
    ndt::type pattern_tp = ndt::type(std::tr1::get<0>(
        ::testing::TestWithParam<std::tr1::tuple<const char *, const char *, const char *>>::GetParam()));
    ndt::type dtp = ndt::type(std::tr1::get<2>(
        ::testing::TestWithParam<std::tr1::tuple<const char *, const char *, const char *>>::GetParam()));

    if (pattern_tp == ndt::type("T")) {
      std::map<std::string, ndt::type> tp_vars;
      tp_vars["R"] = dtp;

      return ndt::substitute(concrete_tp, tp_vars, true);
    }

#ifdef DYND_CUDA
    if (pattern_tp == ndt::type("cuda_device[T]")) {
      std::map<std::string, ndt::type> tp_vars;
      tp_vars["R"] = dtp;

      return ndt::substitute(ndt::make_cuda_device(concrete_tp), tp_vars, true);
    }
#endif

    throw std::runtime_error("error");
  }
};

class FFT2D : public ::testing::TestWithParam<std::tr1::tuple<const char *, const char *, const char *>> {

public:
  ndt::type GetParam() const
  {
    ndt::type concrete_tp = ndt::type(std::tr1::get<1>(
        ::testing::TestWithParam<std::tr1::tuple<const char *, const char *, const char *>>::GetParam()));
    ndt::type pattern_tp = ndt::type(std::tr1::get<0>(
        ::testing::TestWithParam<std::tr1::tuple<const char *, const char *, const char *>>::GetParam()));
    ndt::type dtp = ndt::type(std::tr1::get<2>(
        ::testing::TestWithParam<std::tr1::tuple<const char *, const char *, const char *>>::GetParam()));

    if (pattern_tp == ndt::type("T")) {
      std::map<std::string, ndt::type> tp_vars;
      tp_vars["R"] = dtp;

      return ndt::substitute(concrete_tp, tp_vars, true);
    }

#ifdef DYND_CUDA
    if (pattern_tp == ndt::type("cuda_device[T]")) {
      std::map<std::string, ndt::type> tp_vars;
      tp_vars["R"] = dtp;

      return ndt::substitute(ndt::make_cuda_device(concrete_tp), tp_vars, true);
    }
#endif

    throw std::runtime_error("error");
  }
};

/*
template <typename T>
class FFT1D;

template <typename T, int N>
class FFT1D<dynd::complex<T> [N]> : public ::testing::Test {
public:
  typedef T RealType;
  typedef dynd::complex<T> ComplexType;

  static const intptr_t SrcShape[1];
  static const intptr_t SrcSize;
  typedef ComplexType SrcType;

  static const intptr_t DstShape[1];
  static const intptr_t DstSize;
  typedef ComplexType DstType;
};

template <typename T, int N>
const intptr_t FFT1D<dynd::complex<T> [N]>::SrcShape[1] = {N};

template <typename T, int N>
const intptr_t FFT1D<dynd::complex<T> [N]>::SrcSize = SrcShape[0];

template <typename T, int N>
const intptr_t FFT1D<dynd::complex<T> [N]>::DstShape[1] = {N};

template <typename T, int N>
const intptr_t FFT1D<dynd::complex<T> [N]>::DstSize = DstShape[0];

TYPED_TEST_CASE_P(FFT1D);
*/

template <typename T>
class RFFT1D;

template <typename T, int N>
class RFFT1D<T[N]> : public ::testing::Test {
public:
  typedef T RealType;
  typedef dynd::complex<T> ComplexType;

  static const intptr_t SrcShape[1];
  static const intptr_t SrcSize;
  typedef RealType SrcType;

  static const intptr_t DstShape[1];
  static const intptr_t DstSize;
  typedef ComplexType DstType;
};

template <typename T, int N>
const intptr_t RFFT1D<T[N]>::SrcShape[1] = {N};

template <typename T, int N>
const intptr_t RFFT1D<T[N]>::SrcSize = SrcShape[0];

template <typename T, int N>
const intptr_t RFFT1D<T[N]>::DstShape[1] = {N / 2 + 1};

template <typename T, int N>
const intptr_t RFFT1D<T[N]>::DstSize = DstShape[0];

TYPED_TEST_CASE_P(RFFT1D);

template <typename T>
struct FixedDim1D {
  typedef testing::Types<T[4], T[8], T[17], T[25], T[64], T[76], T[99], T[128], T[203], T[256], T[512]> Types;
};

template <typename T>
class RFFT2D;

template <typename T, int M, int N>
class RFFT2D<T[M][N]> : public ::testing::Test {
public:
  typedef T RealType;
  typedef dynd::complex<T> ComplexType;

  static const intptr_t SrcShape[2];
  static const intptr_t SrcSize;
  typedef RealType SrcType;

  static const intptr_t DstShape[2];
  static const intptr_t DstSize;
  typedef ComplexType DstType;
};

template <typename T, int M, int N>
const intptr_t RFFT2D<T[M][N]>::SrcShape[2] = {M, N};

template <typename T, int M, int N>
const intptr_t RFFT2D<T[M][N]>::SrcSize = SrcShape[0] * SrcShape[1];

template <typename T, int M, int N>
const intptr_t RFFT2D<T[M][N]>::DstShape[2] = {M, N / 2 + 1};

template <typename T, int M, int N>
const intptr_t RFFT2D<T[M][N]>::DstSize = DstShape[0] * DstShape[1];

TYPED_TEST_CASE_P(RFFT2D);

template <typename T>
struct FixedDim2D {
  typedef testing::Types<T[4][4], T[8][8], T[17][25], T[64][64], T[76][14], T[128][128], T[203][99], T[256][256],
                         T[512][512]> Types;
};

template <typename T>
T rel_err_max();

template <>
inline float rel_err_max<float>()
{
  return 1E-4f;
}

template <>
inline double rel_err_max<double>()
{
  return 1E-8;
}

TEST_P(FFT1D, Linear)
{
  const ndt::type &tp = GetParam();

  nd::array x0 = nd::random::uniform({}, {{"dst_tp", tp}});
  nd::array x1 = nd::random::uniform({}, {{"dst_tp", tp}});
  nd::array x = x0 + x1;

  nd::array y0 = nd::fft(x0);
  nd::array y1 = nd::fft(x1);
  nd::array y = nd::fft(x);

#ifdef DYND_CUDA
  y0 = y0.to_host();
  y1 = y1.to_host();
  y = y.to_host();
#endif

  EXPECT_ARRAY_NEAR(y0 + y1, y, rel_err_max<double>());
}

TEST_P(FFT1D, Inverse)
{
  const ndt::type &tp = GetParam();

  nd::array x = nd::random::uniform({}, {{"dst_tp", tp}});

  nd::array y = nd::ifft(nd::fft(x));

#ifdef DYND_CUDA
  x = x.to_host();
  y = y.to_host();
#endif

  EXPECT_ARRAY_NEAR(x, y / y.get_dim_size(), rel_err_max<double>());
}

/*

TYPED_TEST_P(FFT1D, Zeros)
{
  nd::array x = nd::zeros(TestFixture::SrcShape[0],
                          ndt::make_type<typename TestFixture::SrcType>());

  nd::array y = nd::fft(x);

  EXPECT_ARRAY_NEAR(dynd::complex<double>(0.0), y, rel_err_max<double>());
}
*/

/*
TYPED_TEST_P(FFT1D, Ones)
{
  nd::array x = nd::ones(TestFixture::SrcShape[0],
                         ndt::make_type<typename TestFixture::SrcType>());

  nd::array y = nd::fft(x);
  for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
    if (i == 0) {
      EXPECT_EQ_RELERR(TestFixture::SrcSize,
                       y(i).as<typename TestFixture::DstType>(),
                       rel_err_max<typename TestFixture::RealType>());
    } else {
      EXPECT_EQ_RELERR(0, y(i).as<typename TestFixture::DstType>(),
                       rel_err_max<typename TestFixture::RealType>());
    }
  }
}

TYPED_TEST_P(FFT1D, KroneckerDelta)
{
  // Only continue if each dimension is divisible by 2
  if ((TestFixture::SrcShape[0] % 2) != 0) {
    return;
  }

  nd::array x = nd::zeros(TestFixture::SrcShape[0],
                          ndt::make_type<typename TestFixture::SrcType>());
  x(TestFixture::SrcShape[0] / 2).vals() = 1;

  nd::array y = nd::fft(x);

  for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
    EXPECT_EQ_RELERR(pow(static_cast<typename TestFixture::RealType>(-1), i),
                     y(i).as<typename TestFixture::DstType>(),
                     rel_err_max<typename TestFixture::RealType>());
  }

  // Only continue if each dimension is divisible by 4
  if ((TestFixture::SrcShape[0] % 4) != 0) {
    return;
  }

  y = nd::fft(x(irange().by(2)));
  for (int i = 0; i < TestFixture::DstShape[0] / 2; ++i) {
    EXPECT_EQ_RELERR(pow(static_cast<typename TestFixture::RealType>(-1), i),
                     y(i).as<typename TestFixture::DstType>(),
                     rel_err_max<typename TestFixture::RealType>());
  }
}
*/

TEST(FFT1D, Shift)
{
  double vals0[9] = {0.0, 1.0, 2.0, 3.0, 4.0, -4.0, -3.0, -2.0, -1.0};

  nd::array x0 = nd::empty(ndt::make_type<double[9]>());
  x0.vals() = vals0;

  nd::array y0 = nd::fftshift(x0);
  EXPECT_JSON_EQ_ARR("[-4, -3, -2, -1, 0, 1, 2, 3, 4]", y0);

  y0 = nd::ifftshift(y0);
  EXPECT_ARRAY_EQ(x0, y0);

  double vals1[10] = {0.0, 1.0, 2.0, 3.0, 4.0, -5.0, -4.0, -3.0, -2.0, -1.0};

  nd::array x1 = nd::empty(ndt::make_type<double[10]>());
  x1.vals() = vals1;

  nd::array y1 = nd::fftshift(x1);
  EXPECT_JSON_EQ_ARR("[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]", y1);

  y1 = nd::ifftshift(y1);
  EXPECT_ARRAY_EQ(x1, y1);
}

/*
TYPED_TEST_P(RFFT1D, Linear)
{
  nd::array x0 = nd::rand(TestFixture::SrcShape[0],
                          ndt::make_type<typename TestFixture::SrcType>());
  nd::array x1 = nd::rand(TestFixture::SrcShape[0],
                          ndt::make_type<typename TestFixture::SrcType>());
  nd::array x = nd::empty(TestFixture::SrcShape[0],
                          ndt::make_type<typename TestFixture::SrcType>());
  x.vals() = x0 + x1;

  nd::array y0 = nd::rfft(x0);
  nd::array y1 = nd::rfft(x1);
  nd::array y = nd::rfft(x);

  for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
    EXPECT_EQ_RELERR(y0(i).as<typename TestFixture::DstType>() +
                         y1(i).as<typename TestFixture::DstType>(),
                     y(i).as<typename TestFixture::DstType>(),
                     rel_err_max<typename TestFixture::RealType>());
  }
}

TYPED_TEST_P(RFFT1D, Inverse)
{
  nd::array x = nd::rand(TestFixture::SrcShape[0],
                         ndt::make_type<typename TestFixture::SrcType>());

  nd::array y = irfft(nd::rfft(x), TestFixture::SrcShape[0]);
  for (int i = 0; i < TestFixture::SrcShape[0]; ++i) {
    EXPECT_EQ_RELERR(x(i).as<typename TestFixture::SrcType>(),
                     y(i).as<typename TestFixture::SrcType>() /
                         TestFixture::SrcSize,
                     rel_err_max<typename TestFixture::RealType>());
  }
}

TYPED_TEST_P(RFFT1D, Zeros)
{
  nd::array x = nd::zeros(TestFixture::SrcShape[0],
                          ndt::make_type<typename TestFixture::SrcType>());

  nd::array y = nd::rfft(x);
  for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
    EXPECT_EQ(0, y(i).as<typename TestFixture::DstType>());
  }
}

TYPED_TEST_P(RFFT1D, Ones)
{
  nd::array x = nd::ones(TestFixture::SrcShape[0],
                         ndt::make_type<typename TestFixture::SrcType>());

  nd::array y = nd::rfft(x);
  for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
    if (i == 0) {
      EXPECT_EQ_RELERR(TestFixture::SrcSize,
                       y(i).as<typename TestFixture::DstType>(),
                       rel_err_max<typename TestFixture::RealType>());
    } else {
      EXPECT_EQ_RELERR(0, y(i).as<typename TestFixture::DstType>(),
                       rel_err_max<typename TestFixture::RealType>());
    }
  }
}

TYPED_TEST_P(RFFT1D, KroneckerDelta)
{
  // Only continue if each dimension is divisible by 2
  if ((TestFixture::SrcShape[0] % 2) != 0) {
    return;
  }

  nd::array x = nd::zeros(TestFixture::SrcShape[0],
                          ndt::make_type<typename TestFixture::SrcType>());
  x(TestFixture::SrcShape[0] / 2).vals() = 1;

  nd::array y = nd::rfft(x);
  for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
    EXPECT_EQ_RELERR(pow(static_cast<typename TestFixture::RealType>(-1), i),
                     y(i).as<typename TestFixture::DstType>(),
                     rel_err_max<typename TestFixture::RealType>());
  }

  // Only continue if each dimension is divisible by 4
  if ((TestFixture::SrcShape[0] % 4) != 0) {
    return;
  }

  y = nd::rfft(x(irange().by(2)));
  for (int i = 0; i < TestFixture::DstShape[0] / 2; ++i) {
    EXPECT_EQ_RELERR(pow(static_cast<typename TestFixture::RealType>(-1), i),
                     y(i).as<typename TestFixture::DstType>(),
                     rel_err_max<typename TestFixture::RealType>());
  }
}
*/

TEST_P(FFT2D, Linear)
{
  const ndt::type &tp = GetParam();

  nd::array x0 = nd::random::uniform({}, {{"dst_tp", tp}});
  nd::array x1 = nd::random::uniform({}, {{"dst_tp", tp}});
  nd::array x = x0 + x1;

  nd::array y0 = nd::fft(x0);
  nd::array y1 = nd::fft(x1);
  nd::array y = nd::fft(x);

  EXPECT_ARRAY_NEAR(y0 + y1, y, rel_err_max<double>());

  vector<intptr_t> axes;
  axes.push_back(0);

  y0 = nd::fft({x0}, {{"shape", x0.get_shape()}, {"axes", axes}});
  y1 = nd::fft({x1}, {{"shape", x1.get_shape()}, {"axes", axes}});
  y = nd::fft({x}, {{"shape", x.get_shape()}, {"axes", axes}});

  EXPECT_ARRAY_NEAR(y0 + y1, y, rel_err_max<double>());

  if (tp.get_id() != cuda_device_id) {
    axes.clear();
    axes.push_back(1);

    y0 = nd::fft({x0}, {{"shape", x0.get_shape()}, {"axes", axes}});
    y1 = nd::fft({x1}, {{"shape", x1.get_shape()}, {"axes", axes}});
    y = nd::fft({x}, {{"shape", x.get_shape()}, {"axes", axes}});

    EXPECT_ARRAY_NEAR(y0 + y1, y, rel_err_max<double>());
  }
}

TEST_P(FFT2D, Inverse)
{
  const ndt::type &tp = GetParam();

  nd::array x = nd::random::uniform({}, {{"dst_tp", tp}});

  nd::array y = nd::ifft(nd::fft(x));

#ifdef DYND_CUDA
  x = x.to_host();
  y = y.to_host();
#endif

  EXPECT_ARRAY_NEAR(x, y / (y.get_dim_size(0) * y.get_dim_size(1)), rel_err_max<double>());

  /*
  #ifdef DYND_CUDA
    x = x.to_cuda_device();
    y = y.to_cuda_device();
  #endif
  */

  if (tp.get_id() != cuda_device_id) {

    vector<intptr_t> axes;
    axes.push_back(0);

    y = nd::ifft({nd::fft({x}, {{"shape", x.get_shape()}, {"axes", axes}})},
                 {{"shape", y.get_shape()}, {"axes", axes}});

    EXPECT_ARRAY_NEAR(x, y / y.get_dim_size(0), rel_err_max<double>());

    axes.clear();
    axes.push_back(1);

    y = nd::ifft({nd::fft({x}, {{"shape", x.get_shape()}, {"axes", axes}})},
                 {{"shape", y.get_shape()}, {"axes", axes}});

    EXPECT_ARRAY_NEAR(x, y / y.get_dim_size(1), rel_err_max<double>());
  }
}

/*
TYPED_TEST_P(FFT2D, Zeros)
{
  nd::array x = nd::zeros(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
                          ndt::make_type<typename TestFixture::SrcType>());

  nd::array y = nd::fft(x);
  for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
    for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
      EXPECT_EQ(0, y(i, j).as<typename TestFixture::DstType>());
    }
  }

  vector<intptr_t> axes;
  axes.push_back(0);

  y = nd::fft({x}, {{"shape", x.get_shape()}, {"axes", axes}});
  for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
    for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
      EXPECT_EQ(0, y(i, j).as<typename TestFixture::DstType>());
    }
  }

  axes.clear();
  axes.push_back(1);

  y = nd::fft({x}, {{"shape", x.get_shape()}, {"axes", axes}});
  for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
    for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
      EXPECT_EQ(0, y(i, j).as<typename TestFixture::DstType>());
    }
  }
}
*/

/*
TYPED_TEST_P(FFT2D, Ones)
{
  nd::array x = nd::ones(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
                         ndt::make_type<typename TestFixture::SrcType>());

  nd::array y = nd::fft(x);
  for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
    for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
      if (i == 0 && j == 0) {
        EXPECT_EQ_RELERR(TestFixture::SrcSize,
                         y(i, j).as<typename TestFixture::DstType>(),
                         rel_err_max<typename TestFixture::RealType>());
      } else {
        EXPECT_EQ_RELERR(0, y(i, j).as<typename TestFixture::DstType>(),
                         rel_err_max<typename TestFixture::RealType>());
      }
    }
  }
}
*/

/*
TYPED_TEST_P(FFT2D, KroneckerDelta)
{
  // Only continue if each dimension is divisible by 2
  if ((TestFixture::SrcShape[0] % 2) != 0 ||
      (TestFixture::SrcShape[1] % 2) != 0) {
    return;
  }

  nd::array x = nd::zeros(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
                          ndt::make_type<typename TestFixture::SrcType>());
  x(TestFixture::SrcShape[0] / 2, TestFixture::SrcShape[1] / 2).vals() = 1;

  nd::array y = nd::fft(x);
  for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
    for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
      EXPECT_EQ_RELERR(
          pow(static_cast<typename TestFixture::RealType>(-1), i + j),
          y(i, j).as<typename TestFixture::DstType>(),
          rel_err_max<typename TestFixture::RealType>());
    }
  }

  // Only continue if each dimension is divisible by 4
  if ((TestFixture::SrcShape[0] % 4) != 0 ||
      (TestFixture::SrcShape[1] % 4) != 0) {
    return;
  }

  y = nd::fft(x(irange().by(2), irange().by(2)));
  for (int i = 0; i < TestFixture::DstShape[0] / 2; ++i) {
    for (int j = 0; j < TestFixture::DstShape[1] / 2; ++j) {
      EXPECT_EQ_RELERR(
          pow(static_cast<typename TestFixture::RealType>(-1), i + j),
          y(i, j).as<typename TestFixture::DstType>(),
          rel_err_max<typename TestFixture::RealType>());
    }
  }
}
*/

TEST(FFT2D, Shift)
{
  double vals0[3][3] = {{0.0, 1.0, 2.0}, {3.0, 4.0, -4.0}, {-3.0, -2.0, -1.0}};

  nd::array x0 = nd::empty(ndt::make_type<double[3][3]>());
  x0.vals() = vals0;

  nd::array y0 = nd::fftshift(x0);
  EXPECT_EQ(y0(0, 0).as<double>(), -1.0);
  EXPECT_EQ(y0(0, 1).as<double>(), -3.0);
  EXPECT_EQ(y0(0, 2).as<double>(), -2.0);
  EXPECT_EQ(y0(1, 0).as<double>(), 2.0);
  EXPECT_EQ(y0(1, 1).as<double>(), 0.0);
  EXPECT_EQ(y0(1, 2).as<double>(), 1.0);
  EXPECT_EQ(y0(2, 0).as<double>(), -4.0);
  EXPECT_EQ(y0(2, 1).as<double>(), 3.0);
  EXPECT_EQ(y0(2, 2).as<double>(), 4.0);

  y0 = nd::ifftshift(y0);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(y0(i, j).as<double>(), x0(i, j).as<double>());
    }
  }

  double vals1[4][2] = {{0.0, 5.0}, {1.0, 8.0}, {-6.0, 7.0}, {3.0, -1.0}};

  nd::array x1 = nd::empty(ndt::make_type<double[4][2]>());
  x1.vals() = vals1;

  nd::array y1 = nd::fftshift(x1);
  EXPECT_EQ(y1(0, 0).as<double>(), 7.0);
  EXPECT_EQ(y1(0, 1).as<double>(), -6.0);
  EXPECT_EQ(y1(1, 0).as<double>(), -1.0);
  EXPECT_EQ(y1(1, 1).as<double>(), 3.0);
  EXPECT_EQ(y1(2, 0).as<double>(), 5.0);
  EXPECT_EQ(y1(2, 1).as<double>(), 0.0);
  EXPECT_EQ(y1(3, 0).as<double>(), 8.0);
  EXPECT_EQ(y1(3, 1).as<double>(), 1.0);

  y1 = nd::ifftshift(y1);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(y1(i, j).as<double>(), x1(i, j).as<double>());
    }
  }

  double vals2[2][3] = {{0.0, 5.0, 1.0}, {8.0, -6.0, 7.0}};

  nd::array x2 = nd::empty(ndt::make_type<double[2][3]>());
  x2.vals() = vals2;

  nd::array y2 = nd::fftshift(x2);
  EXPECT_EQ(y2(0, 0).as<double>(), 7.0);
  EXPECT_EQ(y2(0, 1).as<double>(), 8.0);
  EXPECT_EQ(y2(0, 2).as<double>(), -6.0);
  EXPECT_EQ(y2(1, 0).as<double>(), 1.0);
  EXPECT_EQ(y2(1, 1).as<double>(), 0.0);
  EXPECT_EQ(y2(1, 2).as<double>(), 5.0);

  y2 = nd::ifftshift(y2);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(y2(i, j).as<double>(), x2(i, j).as<double>());
    }
  }
}

/*
TYPED_TEST_P(RFFT2D, Linear)
{
  nd::array x0 = nd::rand(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
                          ndt::make_type<typename TestFixture::SrcType>());
  nd::array x1 = nd::rand(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
                          ndt::make_type<typename TestFixture::SrcType>());
  nd::array x = nd::empty(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
                          ndt::make_type<typename TestFixture::SrcType>());
  x.vals() = x0 + x1;

  nd::array y0 = nd::rfft(x0);
  nd::array y1 = nd::rfft(x1);
  nd::array y = nd::rfft(x);

  for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
    for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
      EXPECT_EQ_RELERR(y0(i, j).as<typename TestFixture::DstType>() +
                           y1(i, j).as<typename TestFixture::DstType>(),
                       y(i, j).as<typename TestFixture::DstType>(),
                       rel_err_max<typename TestFixture::RealType>());
    }
  }
}

TYPED_TEST_P(RFFT2D, Inverse)
{
  nd::array x = nd::rand(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
                         ndt::make_type<typename TestFixture::SrcType>());

  nd::array y =
      irfft(nd::rfft(x), TestFixture::SrcShape[0], TestFixture::SrcShape[1]);
  for (int i = 0; i < TestFixture::SrcShape[0]; ++i) {
    for (int j = 0; j < TestFixture::SrcShape[1]; ++j) {
      EXPECT_EQ_RELERR(x(i, j).as<typename TestFixture::SrcType>(),
                       y(i, j).as<typename TestFixture::SrcType>() /
                           TestFixture::SrcSize,
                       rel_err_max<typename TestFixture::RealType>());
    }
  }
}

TYPED_TEST_P(RFFT2D, Zeros)
{
  nd::array x = nd::zeros(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
                          ndt::make_type<typename TestFixture::SrcType>());

  nd::array y = nd::rfft(x);
  for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
    for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
      EXPECT_EQ(0, y(i, j).as<typename TestFixture::DstType>());
    }
  }
}

TYPED_TEST_P(RFFT2D, Ones)
{
  nd::array x = nd::ones(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
                         ndt::make_type<typename TestFixture::SrcType>());

  nd::array y = nd::rfft(x);
  for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
    for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
      if (i == 0 && j == 0) {
        EXPECT_EQ_RELERR(TestFixture::SrcSize,
                         y(i, j).as<typename TestFixture::DstType>(),
                         rel_err_max<typename TestFixture::RealType>());
      } else {
        EXPECT_EQ_RELERR(0, y(i, j).as<typename TestFixture::DstType>(),
                         rel_err_max<typename TestFixture::RealType>());
      }
    }
  }
}

TYPED_TEST_P(RFFT2D, KroneckerDelta)
{
  // Only continue if each dimension is divisible by 2
  if ((TestFixture::SrcShape[0] % 2) != 0 ||
      (TestFixture::SrcShape[1] % 2) != 0) {
    return;
  }

  nd::array x = nd::zeros(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
                          ndt::make_type<typename TestFixture::SrcType>());
  x(TestFixture::SrcShape[0] / 2, TestFixture::SrcShape[1] / 2).vals() = 1;

  nd::array y = nd::rfft(x);
  for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
    for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
      EXPECT_EQ_RELERR(
          pow(static_cast<typename TestFixture::RealType>(-1), i + j),
          y(i, j).as<typename TestFixture::DstType>(),
          rel_err_max<typename TestFixture::RealType>());
    }
  }

  // Only continue if each dimension is divisible by 4
  if ((TestFixture::SrcShape[0] % 4) != 0 ||
      (TestFixture::SrcShape[1] % 4) != 0) {
    return;
  }

  y = nd::rfft(x(irange().by(2), irange().by(2)));
  for (int i = 0; i < TestFixture::DstShape[0] / 2; ++i) {
    for (int j = 0; j < TestFixture::DstShape[1] / 2; ++j) {
      EXPECT_EQ_RELERR(
          pow(static_cast<typename TestFixture::RealType>(-1), i + j),
          y(i, j).as<typename TestFixture::DstType>(),
          rel_err_max<typename TestFixture::RealType>());
    }
  }
}
*/

// For now, only test FFTs if we built DYND with FFTW

/** TODO: A few of the single-precision tests fail, even at what should be
 * reasonable relative error.
 *        As all of the double-precision tests are fine, I think this is
 * inherent to FFTW. For now,
 *        I'm commenting out the single-precision tests.
 */

// REGISTER_TYPED_TEST_CASE_P(FFT1D, Linear);
// INSTANTIATE_TYPED_TEST_CASE_P(ComplexFloat, FFT1D,
// FixedDim1D<dynd_complex<float> >::Types);

#define Shapes                                                                                                         \
  ::testing::Values("4 * R", "8 * R", "17 * R", "25 * R", "64 * R", "76 * R", "99 * R", "128 * R", "203 * R", "256 * " \
                                                                                                              "R")
#define DTypes ::testing::Values("complex[float64]")

INSTANTIATE_TEST_CASE_P(Host, FFT1D, ::testing::Combine(::testing::Values("T"), Shapes, DTypes));
#ifdef DYND_CUDA
INSTANTIATE_TEST_CASE_P(CUDADevice, FFT1D, ::testing::Combine(::testing::Values("cuda_device[T]"), Shapes, DTypes));
#endif

#undef Shapes
#undef DTypes

#define Shapes ::testing::Values("4 * 4 * R", "8 * 8 * R", "17 * 25 * R", "64 * 64 * R", "76 * 14 * R")
// "128 * 128 * R", "203 * 99 * R", "256 * 256 * R"

#define DTypes ::testing::Values("complex[float64]")

INSTANTIATE_TEST_CASE_P(Host, FFT2D, ::testing::Combine(::testing::Values("T"), Shapes, DTypes));
#ifdef DYND_CUDA
INSTANTIATE_TEST_CASE_P(CUDADevice, FFT2D, ::testing::Combine(::testing::Values("cuda_device[T]"), Shapes, DTypes));
#endif

#undef Shapes
#undef DTypes

// REGISTER_TYPED_TEST_CASE_P(RFFT1D, Linear, Inverse, Zeros, Ones,
// KroneckerDelta);
// INSTANTIATE_TYPED_TEST_CASE_P(Float, RFFT1D, FixedDim1D<float>::Types);
// INSTANTIATE_TYPED_TEST_CASE_P(ComplexDouble, RFFT1D,
// FixedDim1D<double>::Types);

// REGISTER_TYPED_TEST_CASE_P(RFFT2D, Linear, Inverse, Zeros, Ones,
// KroneckerDelta);
// INSTANTIATE_TYPED_TEST_CASE_P(Float, RFFT2D, FixedDim2D<float>::Types);
// INSTANTIATE_TYPED_TEST_CASE_P(Double, RFFT2D, FixedDim2D<double>::Types);

#endif // DYND_FFTW
