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
#include <dynd/types/strided_dim_type.hpp>

using namespace std;
using namespace std::tr1;
using namespace dynd;


#include <cstdlib>

namespace dynd {
    nd::array rand(int n, const ndt::type& dtp) {
        srand(time(NULL));

        nd::array x = nd::empty(n, ndt::make_strided_dim(dtp));
        for (int i = 0; i < n; i++) {
            x(i).vals() = dynd_complex<double>(std::rand() / ((double) RAND_MAX), std::rand() / ((double) RAND_MAX));
        }

        return x;
    }
}

template <typename T>
class FFT : public ::testing::Test {
public:
    typedef T RealType;
    typedef dynd_complex<T> ComplexType;
};

TYPED_TEST_CASE_P(FFT);

template <typename T>
T rel_err_max();

template <>
inline float rel_err_max<float>() {
    return 1E-4f;
}

template <>
inline double rel_err_max<double>() {
    return 1E-8;
}

const int n = 14;
const size_t sizes[n] = {53, 64, 98, 115, 128, 256, 372, 512, 701, 999, 1024, 1295, 1881, 2048};

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

TEST(FFT, TwoDimDelta) {
    size_t size = 64;

    nd::array x = nd::empty(size, size, ndt::make_strided_dim(ndt::make_strided_dim(ndt::make_type<dynd_complex<double> >())));
    x.vals() = 0;
    x(size / 2, size / 2).vals() = 1;

    nd::array y = fft2(x);

    std::cout << y << std::endl;
}

REGISTER_TYPED_TEST_CASE_P(FFT, OneDimInverse, OneDimZeros, OneDimDelta);

INSTANTIATE_TYPED_TEST_CASE_P(Float, FFT, float);
INSTANTIATE_TYPED_TEST_CASE_P(Double, FFT, double);
