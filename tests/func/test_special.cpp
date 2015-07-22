//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <inc_gtest.hpp>

#include "dynd_assertions.hpp"
#include "special_vals.hpp"

#include <dynd/special.hpp>

using namespace std;
using namespace dynd;

#define REL_ERROR_MAX 1E-11

TEST(Special, Factorial) {
    nd::array vals = factorial_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(vals(i, 1).as<double>(),
                         factorial(vals(i, 0).as<int>()), REL_ERROR_MAX);
    }
}

TEST(Special, Factorial2) {
    nd::array vals = factorial2_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(vals(i, 1).as<double>(),
                         factorial2(vals(i, 0).as<int>()), REL_ERROR_MAX);
    }
}

TEST(Special, FactorialRatio) {
    nd::array vals = factorial_ratio_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(
            vals(i, 2).as<double>(),
            factorial_ratio(vals(i, 0).as<int>(), vals(i, 1).as<int>()),
            REL_ERROR_MAX);
    }
}

TEST(Special, Gamma) {
    using dynd::gamma;

    nd::array vals = gamma_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(vals(i, 1).as<double>(),
                         gamma(vals(i, 0).as<double>()), REL_ERROR_MAX);
    }
}

TEST(Special, LogGamma) {
    using dynd::lgamma;

    nd::array vals = lgamma_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(vals(i, 1).as<double>(),
                         lgamma(vals(i, 0).as<double>()), REL_ERROR_MAX);
    }
}

TEST(Special, Airy) {
    nd::array vals = airy_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        double res[2][2];
        airy(res, vals(i, 0).as<double>());

        EXPECT_EQ_RELERR(vals(i, 1, 0, 0).as<double>(), res[0][0],
                         REL_ERROR_MAX);
        EXPECT_EQ_RELERR(vals(i, 1, 0, 1).as<double>(), res[0][1],
                         REL_ERROR_MAX);
        EXPECT_EQ_RELERR(vals(i, 1, 1, 0).as<double>(), res[1][0],
                         REL_ERROR_MAX);
        EXPECT_EQ_RELERR(vals(i, 1, 1, 1).as<double>(), res[1][1],
                         REL_ERROR_MAX);
    }
}

TEST(Special, BesselJ0) {
    nd::array vals = bessel_j0_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(vals(i, 1).as<double>(),
                         bessel_j0(vals(i, 0).as<double>()), REL_ERROR_MAX);
    }
}

TEST(Special, BesselJ1) {
    nd::array vals = bessel_j1_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(vals(i, 1).as<double>(),
                         bessel_j1(vals(i, 0).as<double>()), REL_ERROR_MAX);
    }
}

TEST(Special, BesselJ) {
    nd::array vals = bessel_j_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(
            vals(i, 2).as<double>(),
            bessel_j(vals(i, 0).as<double>(), vals(i, 1).as<double>()),
            REL_ERROR_MAX);
    }
}

TEST(Special, SphericalBesselJ0) {
    nd::array vals = sph_bessel_j0_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(vals(i, 1).as<double>(),
                         sph_bessel_j0(vals(i, 0).as<double>()), REL_ERROR_MAX);
    }
}

TEST(Special, SphericalBesselJ) {
    nd::array vals = sph_bessel_j_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(
            vals(i, 2).as<double>(),
            sph_bessel_j(vals(i, 0).as<double>(), vals(i, 1).as<double>()),
            REL_ERROR_MAX);
    }
}

TEST(Special, BesselY0) {
    nd::array vals = bessel_y0_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(vals(i, 1).as<double>(),
                         bessel_y0(vals(i, 0).as<double>()), REL_ERROR_MAX);
    }
}

TEST(Special, BesselY1) {
    nd::array vals = bessel_y1_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(vals(i, 1).as<double>(),
                         bessel_y1(vals(i, 0).as<double>()), REL_ERROR_MAX);
    }
}

TEST(Special, BesselY) {
    nd::array vals = bessel_y_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(
            vals(i, 2).as<double>(),
            bessel_y(vals(i, 0).as<double>(), vals(i, 1).as<double>()),
            REL_ERROR_MAX);
    }
}

TEST(Special, SphericalBesselY0) {
    nd::array vals = sph_bessel_y0_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(vals(i, 1).as<double>(),
                         sph_bessel_y0(vals(i, 0).as<double>()), REL_ERROR_MAX);
    }
}

TEST(Special, SphericalBesselY) {
    nd::array vals = sph_bessel_y_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(
            vals(i, 2).as<double>(),
            sph_bessel_y(vals(i, 0).as<double>(), vals(i, 1).as<double>()),
            REL_ERROR_MAX);
    }
}

TEST(Special, HankelH1) {
    nd::array vals = hankel_h1_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(
            vals(i, 2).as<dynd::complex<double> >(),
            hankel_h1(vals(i, 0).as<double>(), vals(i, 1).as<double>()),
            REL_ERROR_MAX);
    }
}

TEST(Special, SphericalHankelH1) {
    nd::array vals = sph_hankel_h1_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(
            vals(i, 2).as<dynd::complex<double> >(),
            sph_hankel_h1(vals(i, 0).as<double>(), vals(i, 1).as<double>()),
            REL_ERROR_MAX);
    }
}

TEST(Special, HankelH2) {
    nd::array vals = hankel_h2_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(
            vals(i, 2).as<dynd::complex<double> >(),
            hankel_h2(vals(i, 0).as<double>(), vals(i, 1).as<double>()),
            REL_ERROR_MAX);
    }
}

TEST(Special, SphericalHankelH2) {
    nd::array vals = sph_hankel_h2_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(
            vals(i, 2).as<dynd::complex<double> >(),
            sph_hankel_h2(vals(i, 0).as<double>(), vals(i, 1).as<double>()),
            REL_ERROR_MAX);
    }
}

TEST(Special, StruveH) {
    nd::array vals = struve_h_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(
            vals(i, 2).as<double>(),
            struve_h(vals(i, 0).as<double>(), vals(i, 1).as<double>()),
            REL_ERROR_MAX);
    }
}

TEST(Special, LegendreP) {
    nd::array vals = legendre_p_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(
            vals(i, 2).as<double>(),
            legendre_p(vals(i, 0).as<int>(), vals(i, 1).as<double>()),
            REL_ERROR_MAX);
    }
}

TEST(Special, AssociatedLegendreP) {
    nd::array vals = assoc_legendre_p_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_EQ_RELERR(vals(i, 3).as<double>(),
                         assoc_legendre_p(vals(i, 0).as<int>(),
                                          vals(i, 1).as<int>(),
                                          vals(i, 2).as<double>()),
                         REL_ERROR_MAX);
    }
}

#undef REL_ERROR_MAX
