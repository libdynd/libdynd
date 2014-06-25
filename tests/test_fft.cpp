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

    static const bool Redundant;

    static const intptr_t SrcShape[2];
    static const intptr_t SrcSize;
    typedef RealType SrcType;

    static const intptr_t DstShape[2];
    static const intptr_t DstSize;
    typedef ComplexType DstType;
};

template <typename T, int M, int N>
class FFT2D<dynd_complex<T>[M][N]> : public ::testing::Test {
public:
    typedef T RealType;
    typedef dynd_complex<T> ComplexType;

    static const bool Redundant;

    static const intptr_t SrcShape[2];
    static const intptr_t SrcSize;
    typedef ComplexType SrcType;

    static const intptr_t DstShape[2];
    static const intptr_t DstSize;
    typedef ComplexType DstType;
};

template <typename T, int M, int N>
const intptr_t FFT2D<T[M][N]>::SrcShape[2] = {M, N};

template <typename T, int M, int N>
const bool FFT2D<T[M][N]>::Redundant = false;

template <typename T, int M, int N>
const intptr_t FFT2D<T[M][N]>::SrcSize = M * N; 

template <typename T, int M, int N>
const intptr_t FFT2D<T[M][N]>::DstShape[2] = {M, N / 2 + 1};

template <typename T, int M, int N>
const intptr_t FFT2D<T[M][N]>::DstSize = M * (N / 2 + 1); 

template <typename T, int M, int N>
const intptr_t FFT2D<dynd_complex<T>[M][N]>::SrcShape[2] = {M, N}; 

template <typename T, int M, int N>
const intptr_t FFT2D<dynd_complex<T>[M][N]>::SrcSize = M * N; 

template <typename T, int M, int N>
const intptr_t FFT2D<dynd_complex<T>[M][N]>::DstShape[2] = {M, N};

template <typename T, int M, int N>
const intptr_t FFT2D<dynd_complex<T>[M][N]>::DstSize = M * N;

template <typename T, int M, int N>
const bool FFT2D<dynd_complex<T>[M][N]>::Redundant = true;

template <typename T>
struct FixedDim2D {
    typedef testing::Types<T[4][4], T[8][8], T[17][25], T[64][64], T[76][14], T[128][128],
        T[203][99], T[256][256], T[512][512]> Types;    

    typedef testing::Types<T[4][4]> SimpleTypes;    
};

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



/*
TYPED_TEST_P(FFT2D, Linear) {
    nd::array x0 = dynd::rand(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());
    nd::array x1 = dynd::rand(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());
    nd::array x = x0 + x1;

  //  nd::array y0 = fft2(x0);
//    nd::array y1 = fft2(x1);
//    nd::array y = fft2(x);

    for (int i = 0; i < TestFixture::SrcShape[0]; ++i) {
        for (int j = 0; j < TestFixture::SrcShape[1]; ++j) {
            EXPECT_EQ_RELERR(y0(i, j).as<typename TestFixture::DstType>() + y1(i, j).as<typename TestFixture::DstType>(),
                y(i, j).as<typename TestFixture::DstType>(),
                rel_err_max<typename TestFixture::RealType>());
        }
    }
}
*/

TYPED_TEST_P(FFT2D, Inverse) {
    nd::array x = dynd::rand(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());

    nd::array y = ifft2(fft2(x, TestFixture::Redundant), TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        TestFixture::Redundant);
    for (int i = 0; i < TestFixture::SrcShape[0]; ++i) {
        for (int j = 0; j < TestFixture::SrcShape[1]; ++j) {
            EXPECT_EQ_RELERR(x(i, j).as<typename TestFixture::SrcType>(),
                y(i, j).as<typename TestFixture::SrcType>() / TestFixture::SrcSize,
                rel_err_max<typename TestFixture::RealType>());
        }
    }
}

TYPED_TEST_P(FFT2D, Zeros) {
    nd::array x = nd::zeros(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_strided_dim(ndt::make_strided_dim(ndt::make_type<typename TestFixture::SrcType>())));

    nd::array y = fft2(x, TestFixture::Redundant);
    for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
        for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
            EXPECT_EQ(0, y(i, j).as<typename TestFixture::DstType>());
        }
    }
}

TYPED_TEST_P(FFT2D, Ones) {
    nd::array x = nd::ones(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_strided_dim(ndt::make_strided_dim(ndt::make_type<typename TestFixture::SrcType>())));

    nd::array y = fft2(x, TestFixture::Redundant);
    for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
        for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
            if (i == 0 && j == 0) {
                EXPECT_EQ_RELERR(TestFixture::SrcSize, y(i, j).as<typename TestFixture::DstType>(),
                    rel_err_max<typename TestFixture::RealType>());
            } else {
                EXPECT_EQ_RELERR(0, y(i, j).as<typename TestFixture::DstType>(),
                    rel_err_max<typename TestFixture::RealType>());
            }
        }
    }
}

TYPED_TEST_P(FFT2D, KroneckerDelta) {
    if ((TestFixture::SrcShape[0] % 2) != 0 || (TestFixture::SrcShape[1] % 2) != 0) {
        return;
    }

    nd::array x = nd::zeros(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_strided_dim(ndt::make_strided_dim(ndt::make_type<typename TestFixture::SrcType>())));
    x(TestFixture::SrcShape[0] / 2, TestFixture::SrcShape[1] / 2).vals() = 1;

    nd::array y = fft2(x, false);
    for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
        for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
            EXPECT_EQ(pow(-1.0, i) * pow(-1.0, j), y(i, j).as<typename TestFixture::DstType>());
        }
    }

/*
    if ((TestFixture::SrcShape[0] % 4) != 0 || (TestFixture::SrcShape[1] % 4) != 0) {
        return;
    }

    y = fft2(x(irange().by(2), irange().by(2)));
    for (int i = 0; i < TestFixture::SrcShape[0] / 2; ++i) {
        for (int j = 0; j < TestFixture::SrcShape[1] / 2; ++j) {
            EXPECT_EQ(pow(-1.0, i) * pow(-1.0, j), y(i, j).as<typename TestFixture::DstType>());
        }
    }
*/
}

TEST(FFT2D, Shift)  {
    static int vals[3][3] = {{0, 1, 2}, {3, 4, -4}, {-3, -2, -1}};

    nd::array x = nd::empty<int[3][3]>();
    x.vals() = vals;

    nd::array y = fftshift(x);
}



/*
    if ((TestFixture::SrcShape[0] % 4) != 0 || (TestFixture::SrcShape[1] % 4) != 0) {
        return;
    }

    y = rfft2(x(irange().by(2), irange().by(2)));
    for (int i = 0; i < TestFixture::DstShape[0] / 2; ++i) {
        for (int j = 0; j < TestFixture::DstShape[1] / 2; ++j) {
            EXPECT_EQ(pow(-1.0, i) * pow(-1.0, j), y(i, j).as<typename TestFixture::DstType>());
        }
    }
*/

//REGISTER_TYPED_TEST_CASE_P(FFT2D, Inverse, Zeros, Ones, KroneckerDelta);
//INSTANTIATE_TYPED_TEST_CASE_P(ComplexDouble, FFT2D, FixedDim2D<dynd_complex<double> >::Types);

//INSTANTIATE_TYPED_TEST_CASE_P(ComplexFloat, FFT2D, FixedDim2D<dynd_complex<float> >::Types);
//INSTANTIATE_TYPED_TEST_CASE_P(Double, FFT2D, FixedDim2D<double>::Types);
