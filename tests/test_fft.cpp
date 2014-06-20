//
// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
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
#include <dynd/types/strided_dim_type.hpp>

using namespace std;
using namespace dynd;

template <typename T>
class FFT2D;

template <typename T, int M, int N>
class FFT2D<T[M][N]> : public ::testing::Test {
public:
    typedef T RealType;
    typedef dynd_complex<T> ComplexType;

    typedef RealType SrcType;
    typedef ComplexType DstType;

    static const intptr_t Shape[2];
    static const intptr_t Size = M * N;
};

template <typename T, int M, int N>
class FFT2D<dynd_complex<T>[M][N]> : public ::testing::Test {
public:
    typedef T RealType;
    typedef dynd_complex<T> ComplexType;

    typedef ComplexType SrcType;
    typedef ComplexType DstType;

    static const intptr_t Shape[2];
    static const intptr_t Size;

};

template <typename T, int M, int N>
const intptr_t FFT2D<T[M][N]>::Shape[2] = {M, N}; 

template <typename T, int M, int N>
const intptr_t FFT2D<dynd_complex<T>[M][N]>::Shape[2] = {M, N}; 

template <typename T, int M, int N>
const intptr_t FFT2D<dynd_complex<T>[M][N]>::Size = M * N; 

typedef testing::Types<dynd_complex<float>[4][4], dynd_complex<float>[8][8], dynd_complex<float>[17][25],
    dynd_complex<float>[64][64], dynd_complex<float>[76][14], dynd_complex<float>[128][128],
    dynd_complex<float>[203][99], dynd_complex<float>[256][256], dynd_complex<float>[512][512]> FloatFFT2DTypes;
typedef testing::Types<dynd_complex<double>[4][4], dynd_complex<double>[8][8], dynd_complex<double>[17][25],
    dynd_complex<double>[64][64], dynd_complex<double>[76][14], dynd_complex<double>[128][128],
    dynd_complex<double>[203][99], dynd_complex<double>[256][256], dynd_complex<double>[512][512]> DoubleFFT2DTypes;

TYPED_TEST_CASE_P(FFT2D);

template <typename T>
T rel_err_max();

template <>
inline float rel_err_max<float>() {
    return 1E-3f;
}

template <>
inline double rel_err_max<double>() {
    return 1E-8;
}

const int n = 14;
const size_t sizes[n] = {53, 64, 98, 115, 128, 256, 372, 512, 701, 999, 1024, 1295, 1881, 2048};

/*
TYPED_TEST_P(FFT, OneDimInverse) {
    typedef typename TestFixture::RealType real_type;
    typedef typename TestFixture::ComplexType complex_type;

    for (int i = 0; i < n; ++i) {
        size_t size = sizes[i];

        nd::array x = dynd::rand(size, ndt::make_type<complex_type>());
        nd::array y = ifft1(fft1(x));

        EXPECT_EQ(x.get_dim_size(), y.get_dim_size());
        for (int i = 0; i < y.get_dim_size(); ++i) {
            EXPECT_EQ_RELERR(x(i).as<complex_type>(), y(i).as<complex_type>() / size, rel_err_max<real_type>());
        }
    }
}

TYPED_TEST_P(FFT, OneDimZeros) {
    typedef typename TestFixture::RealType real_type;
    typedef typename TestFixture::ComplexType complex_type;

    for (int i = 0; i < n; ++i) {
        size_t size = sizes[i];

        nd::array x = nd::empty(size, ndt::make_strided_dim(ndt::make_type<complex_type>()));
        x.vals() = 0.0;
        nd::array y = fft1(x);

        EXPECT_EQ(x.get_dim_size(), y.get_dim_size());
        for (int i = 0; i < y.get_dim_size(); ++i) {
            EXPECT_EQ_RELERR(complex_type(0, 0), y(i).as<complex_type>(), rel_err_max<real_type>());
        }
    }
}

TEST(FFT, OneDimOnes) {
}

TYPED_TEST_P(FFT, OneDimDelta) {
    typedef typename TestFixture::RealType real_type;
    typedef typename TestFixture::ComplexType complex_type;

    for (int s = 0; s < n; ++s) {
        size_t size = sizes[s];
        if (size != 64 && size != 128 && size != 256 && size != 512 && size != 1024 && size != 2048) {
            continue;
        }

        nd::array x = nd::empty(size, ndt::make_strided_dim(ndt::make_type<complex_type>()));
        x.vals() = 0;
        x(x.get_dim_size() / 2).vals() = 1;

        nd::array y = fft1(x);
        EXPECT_EQ(x.get_dim_size(), y.get_dim_size());
        for (int i = 0; i < x.get_dim_size(); i += 2) {
            EXPECT_EQ(1.0, y(i).as<complex_type>().real());
            EXPECT_EQ(-1.0, y(i + 1).as<complex_type>().real());
        }

        nd::array xe = x(irange().by(2));
        xe.vals() = 0;
        xe(xe.get_dim_size() / 2).vals() = 1;

        nd::array xo = x(1 <= irange().by(2));
        xo.vals() = dynd::rand(xo.get_dim_size(), ndt::make_type<complex_type>());

        y = fft1(xe);
        for (int i = 0; i < y.get_dim_size(); i += 2) {
            EXPECT_EQ(1.0, y(i).as<complex_type>().real());
            EXPECT_EQ(-1.0, y(i + 1).as<complex_type>().real());
        }
    }
}
*/

TYPED_TEST_P(FFT2D, Inverse) {
    nd::array x = dynd::rand(TestFixture::Shape[0], TestFixture::Shape[1],
        ndt::make_type<typename TestFixture::SrcType>());

    nd::array y = ifft2(fft2(x));

    for (int i = 0; i < TestFixture::Shape[0]; ++i) {
        for (int j = 0; j < TestFixture::Shape[1]; ++j) {
            EXPECT_EQ_RELERR(x(i, j).as<typename TestFixture::DstType>(),
                y(i, j).as<typename TestFixture::DstType>() / TestFixture::Size,
                rel_err_max<typename TestFixture::RealType>());
        }
    }
}

/*
TYPED_TEST_P(FFT2D, Linear) {
    nd::array x0 = dynd::rand(TestFixture::Shape[0], TestFixture::Shape[1],
        ndt::make_type<typename TestFixture::SrcType>());
    nd::array x1 = dynd::rand(TestFixture::Shape[0], TestFixture::Shape[1],
        ndt::make_type<typename TestFixture::SrcType>());
    nd::array x = x0 + x1;

  //  nd::array y0 = fft2(x0);
//    nd::array y1 = fft2(x1);
//    nd::array y = fft2(x);

    for (int i = 0; i < TestFixture::Shape[0]; ++i) {
        for (int j = 0; j < TestFixture::Shape[1]; ++j) {
            EXPECT_EQ_RELERR(y0(i, j).as<typename TestFixture::DstType>() + y1(i, j).as<typename TestFixture::DstType>(),
                y(i, j).as<typename TestFixture::DstType>(),
                rel_err_max<typename TestFixture::RealType>());
        }
    }
}
*/

TYPED_TEST_P(FFT2D, Zeros) {
    nd::array x = nd::zeros(TestFixture::Shape[0], TestFixture::Shape[1],
        ndt::make_strided_dim(ndt::make_strided_dim(ndt::make_type<typename TestFixture::SrcType>())));

    nd::array y = fft2(x);
    for (int i = 0; i < TestFixture::Shape[0]; ++i) {
        for (int j = 0; j < TestFixture::Shape[1]; ++j) {
            EXPECT_EQ(0, y(i, j).as<typename TestFixture::DstType>());
        }
    }

    y = ifft2(x);
    for (int i = 0; i < TestFixture::Shape[0]; ++i) {
        for (int j = 0; j < TestFixture::Shape[1]; ++j) {
            EXPECT_EQ(0, y(i, j).as<typename TestFixture::DstType>());
        }
    }
}

TYPED_TEST_P(FFT2D, Ones) {
    nd::array x = nd::ones(TestFixture::Shape[0], TestFixture::Shape[1],
        ndt::make_strided_dim(ndt::make_strided_dim(ndt::make_type<typename TestFixture::SrcType>())));

    nd::array y = fft2(x);
    for (int i = 0; i < TestFixture::Shape[0]; ++i) {
        for (int j = 0; j < TestFixture::Shape[1]; ++j) {
            if (i == 0 && j == 0) {
                EXPECT_EQ_RELERR(TestFixture::Size, y(i, j).as<typename TestFixture::DstType>(),
                    rel_err_max<typename TestFixture::RealType>());
            } else {
                EXPECT_EQ_RELERR(0, y(i, j).as<typename TestFixture::DstType>(),
                    rel_err_max<typename TestFixture::RealType>());
            }
        }
    }

    y = ifft2(x);
    for (int i = 0; i < TestFixture::Shape[0]; ++i) {
        for (int j = 0; j < TestFixture::Shape[1]; ++j) {
            if (i == 0 && j == 0) {
                EXPECT_EQ_RELERR(TestFixture::Size, y(i, j).as<typename TestFixture::DstType>(),
                    rel_err_max<typename TestFixture::RealType>());
            } else {
                EXPECT_EQ_RELERR(0, y(i, j).as<typename TestFixture::DstType>(),
                    rel_err_max<typename TestFixture::RealType>());
            }
        }
    }
}

TYPED_TEST_P(FFT2D, KroneckerDelta) {
    if (TestFixture::Shape[0] % 2 != 0 || TestFixture::Shape[1] % 2 != 0) {
        return;
    }

    nd::array x = nd::zeros(TestFixture::Shape[0], TestFixture::Shape[1],
        ndt::make_strided_dim(ndt::make_strided_dim(ndt::make_type<typename TestFixture::SrcType>())));
    x(TestFixture::Shape[0] / 2, TestFixture::Shape[1] / 2).vals() = 1;

    nd::array y = fft2(x);
    for (int i = 0; i < TestFixture::Shape[0]; ++i) {
        for (int j = 0; j < TestFixture::Shape[1]; ++j) {
            EXPECT_EQ(pow(-1.0, i) * pow(-1.0, j), y(i, j).as<typename TestFixture::DstType>());
        }
    }
}

REGISTER_TYPED_TEST_CASE_P(FFT2D, Inverse, Zeros, Ones, KroneckerDelta);

INSTANTIATE_TYPED_TEST_CASE_P(Float, FFT2D, FloatFFT2DTypes);
INSTANTIATE_TYPED_TEST_CASE_P(Double, FFT2D, DoubleFFT2DTypes);
