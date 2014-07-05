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

#include <dynd/array_range.hpp>
#include <dynd/fft.hpp>
#include <dynd/random.hpp>

using namespace std;
using namespace dynd;

#ifdef DYND_FFTW // For now, only test FFTs if we built DYND with FFTW

template <typename T>
class FFT2D;

template <typename T, int M, int N>
class FFT2D<dynd_complex<T>[M][N]> : public ::testing::Test {
public:
    typedef T RealType;
    typedef dynd_complex<T> ComplexType;

    static const intptr_t SrcShape[2];
    static const intptr_t SrcSize;
    typedef ComplexType SrcType;

    static const intptr_t DstShape[2];
    static const intptr_t DstSize;
    typedef ComplexType DstType;
};

template <typename T, int M, int N>
const intptr_t FFT2D<dynd_complex<T>[M][N]>::SrcShape[2] = {M, N};

template <typename T, int M, int N>
const intptr_t FFT2D<dynd_complex<T>[M][N]>::SrcSize = SrcShape[0] * SrcShape[1];

template <typename T, int M, int N>
const intptr_t FFT2D<dynd_complex<T>[M][N]>::DstShape[2] = {M, N};

template <typename T, int M, int N>
const intptr_t FFT2D<dynd_complex<T>[M][N]>::DstSize = DstShape[0] * DstShape[1];

TYPED_TEST_CASE_P(FFT2D);

template <typename T>
class RFFT2D;

template <typename T, int M, int N>
class RFFT2D<T[M][N]> : public ::testing::Test {
public:
    typedef T RealType;
    typedef dynd_complex<T> ComplexType;

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
    typedef testing::Types<T[4][4], T[8][8], T[17][25], T[64][64], T[76][14], T[128][128],
        T[203][99], T[256][256], T[512][512]> Types;    
};

template <typename T>
T rel_err_max();

template <>
inline float rel_err_max<float>() {
    return 1E-4;
}

template <>
inline double rel_err_max<double>() {
    return 1E-8;
}

TYPED_TEST_P(FFT2D, Linear) {
    nd::array x0 = nd::dtyped_rand(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());
    nd::array x1 = nd::dtyped_rand(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());
    nd::array x = nd::dtyped_empty(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());
    x.vals() = x0 + x1;

    nd::array y0 = fft(x0);
    nd::array y1 = fft(x1);
    nd::array y = fft(x);

    for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
        for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
            EXPECT_EQ_RELERR(y0(i, j).as<typename TestFixture::DstType>() + y1(i, j).as<typename TestFixture::DstType>(),
                y(i, j).as<typename TestFixture::DstType>(),
                rel_err_max<typename TestFixture::RealType>());
        }
    }
}

TYPED_TEST_P(FFT2D, Inverse) {
    nd::array x = nd::dtyped_rand(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());

    nd::array y = ifft(fft(x));
    for (int i = 0; i < TestFixture::SrcShape[0]; ++i) {
        for (int j = 0; j < TestFixture::SrcShape[1]; ++j) {
            EXPECT_EQ_RELERR(x(i, j).as<typename TestFixture::SrcType>(),
                y(i, j).as<typename TestFixture::SrcType>() / TestFixture::SrcSize,
                rel_err_max<typename TestFixture::RealType>());
        }
    }
}

TYPED_TEST_P(FFT2D, Zeros) {
    nd::array x = nd::dtyped_zeros(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());

    nd::array y = fft(x);
    for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
        for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
            EXPECT_EQ(0, y(i, j).as<typename TestFixture::DstType>());
        }
    }
}

TYPED_TEST_P(FFT2D, Ones) {
    nd::array x = nd::dtyped_ones(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());

    nd::array y = fft(x);
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
    // Only continue if each dimension is divisible by 2
    if ((TestFixture::SrcShape[0] % 2) != 0 || (TestFixture::SrcShape[1] % 2) != 0) {
        return;
    }

    nd::array x = nd::dtyped_zeros(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());
    x(TestFixture::SrcShape[0] / 2, TestFixture::SrcShape[1] / 2).vals() = 1;

    nd::array y = fft(x);
    for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
        for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
            EXPECT_EQ(pow(static_cast<typename TestFixture::RealType>(-1), i + j),
                y(i, j).as<typename TestFixture::DstType>());
        }
    }

    // Only continue if each dimension is divisible by 4
    if ((TestFixture::SrcShape[0] % 4) != 0 || (TestFixture::SrcShape[1] % 4) != 0) {
        return;
    }

    y = fft(x(irange().by(2), irange().by(2)));
    for (int i = 0; i < TestFixture::DstShape[0] / 2; ++i) {
        for (int j = 0; j < TestFixture::DstShape[1] / 2; ++j) {
            EXPECT_EQ(pow(static_cast<typename TestFixture::RealType>(-1), i + j),
                y(i, j).as<typename TestFixture::DstType>());
        }
    }
}

TYPED_TEST_P(RFFT2D, Linear) {
    nd::array x0 = nd::dtyped_rand(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());
    nd::array x1 = nd::dtyped_rand(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());
    nd::array x = nd::dtyped_empty(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());
    x.vals() = x0 + x1;

    nd::array y0 = rfft(x0);
    nd::array y1 = rfft(x1);
    nd::array y = rfft(x);

    for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
        for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
            EXPECT_EQ_RELERR(y0(i, j).as<typename TestFixture::DstType>() + y1(i, j).as<typename TestFixture::DstType>(),
                y(i, j).as<typename TestFixture::DstType>(),
                rel_err_max<typename TestFixture::RealType>());
        }
    }
}

TYPED_TEST_P(RFFT2D, Inverse) {
    nd::array x = nd::dtyped_rand(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());

    nd::array y = irfft(rfft(x), TestFixture::SrcShape[0], TestFixture::SrcShape[1]);
    for (int i = 0; i < TestFixture::SrcShape[0]; ++i) {
        for (int j = 0; j < TestFixture::SrcShape[1]; ++j) {
            EXPECT_EQ_RELERR(x(i, j).as<typename TestFixture::SrcType>(),
                y(i, j).as<typename TestFixture::SrcType>() / TestFixture::SrcSize,
                rel_err_max<typename TestFixture::RealType>());
        }
    }
}

TYPED_TEST_P(RFFT2D, Zeros) {
    nd::array x = nd::dtyped_zeros(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());

    nd::array y = rfft(x);
    for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
        for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
            EXPECT_EQ(0, y(i, j).as<typename TestFixture::DstType>());
        }
    }
}

TYPED_TEST_P(RFFT2D, Ones) {
    nd::array x = nd::dtyped_ones(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());

    nd::array y = rfft(x);
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

TYPED_TEST_P(RFFT2D, KroneckerDelta) {
    // Only continue if each dimension is divisible by 2
    if ((TestFixture::SrcShape[0] % 2) != 0 || (TestFixture::SrcShape[1] % 2) != 0) {
        return;
    }

    nd::array x = nd::dtyped_zeros(TestFixture::SrcShape[0], TestFixture::SrcShape[1],
        ndt::make_type<typename TestFixture::SrcType>());
    x(TestFixture::SrcShape[0] / 2, TestFixture::SrcShape[1] / 2).vals() = 1;

    nd::array y = rfft(x);
    for (int i = 0; i < TestFixture::DstShape[0]; ++i) {
        for (int j = 0; j < TestFixture::DstShape[1]; ++j) {
            EXPECT_EQ(pow(static_cast<typename TestFixture::RealType>(-1), i + j),
                y(i, j).as<typename TestFixture::DstType>());
        }
    }

    // Only continue if each dimension is divisible by 4
    if ((TestFixture::SrcShape[0] % 4) != 0 || (TestFixture::SrcShape[1] % 4) != 0) {
        return;
    }

    y = rfft(x(irange().by(2), irange().by(2)));
    for (int i = 0; i < TestFixture::DstShape[0] / 2; ++i) {
        for (int j = 0; j < TestFixture::DstShape[1] / 2; ++j) {
            EXPECT_EQ(pow(static_cast<typename TestFixture::RealType>(-1), i + j),
                y(i, j).as<typename TestFixture::DstType>());
        }
    }
}

TEST(FFT2D, Shift)  {
    static int vals[3][3] = {{0, 1, 2}, {3, 4, -4}, {-3, -2, -1}};
    nd::array x = nd::empty<int[3][3]>();
    x.vals() = vals;

    nd::array y = fftshift(x);
    std::cout << "(DEBUG) " << y << std::endl;

    nd::array z = ifftshift(y);
    std::cout << "(DEBUG) " << z << std::endl;
}

REGISTER_TYPED_TEST_CASE_P(FFT2D, Linear, Inverse, Zeros, Ones, KroneckerDelta);
// INSTANTIATE_TYPED_TEST_CASE_P(ComplexFloat, FFT2D, FixedDim2D<dynd_complex<float> >::Types);
INSTANTIATE_TYPED_TEST_CASE_P(ComplexDouble, FFT2D, FixedDim2D<dynd_complex<double> >::Types);

REGISTER_TYPED_TEST_CASE_P(RFFT2D, Linear, Inverse, Zeros, Ones, KroneckerDelta);
// INSTANTIATE_TYPED_TEST_CASE_P(Float, RFFT2D, FixedDim2D<float>::Types);
INSTANTIATE_TYPED_TEST_CASE_P(Double, RFFT2D, FixedDim2D<double>::Types);

#endif // DYND_FFTW
