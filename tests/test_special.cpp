// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <inc_gtest.hpp>

#include <dynd/special.hpp>

#include "special_vals.hpp"

using namespace std;
using namespace dynd;

double rel_error(double expected, double actual) {
    if ((expected == 0.0) && (actual == 0.0)) {
        return 0.0;
    }

    return fabs(1.0 - actual / expected);
}

#define REL_ERROR_MAX 1E-8

TEST(Special, Factorial) {
    nd::array vals = factorial_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(i, 1).as<double>(), factorial(vals(i, 0).as<int>())));
    }
}

TEST(Special, Factorial2) {
    nd::array vals = factorial2_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(i, 1).as<double>(), factorial2(vals(i, 0).as<int>())));
    }
}

TEST(Special, FactorialRatio) {
    nd::array vals = factorial_ratio_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(i, 2).as<double>(), factorial_ratio(vals(i, 0).as<int>(), vals(i, 1).as<int>())));
    }
}

TEST(Special, Gamma) {
    using dynd::gamma;

    nd::array vals = gamma_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(i, 1).as<double>(), gamma(vals(i, 0).as<double>())));
    }
}

TEST(Special, LogGamma) {
    using dynd::lgamma;

    nd::array vals = lgamma_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(i, 1).as<double>(), lgamma(vals(i, 0).as<double>())));
    }
}

TEST(Special, BesselJ0) {
    nd::array vals = bessel_j0_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(i, 1).as<double>(), bessel_j0(vals(i, 0).as<double>())));
    }
}

TEST(Special, BesselJ1) {
    nd::array vals = bessel_j1_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(i, 1).as<double>(), bessel_j1(vals(i, 0).as<double>())));
    }
}

TEST(Special, BesselJ) {
    nd::array vals = bessel_j_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(i, 2).as<double>(), bessel_j(vals(i, 0).as<double>(), vals(i, 1).as<double>())));
    }
}

TEST(Special, SphericalBesselJ0) {
    nd::array vals = sph_bessel_j0_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(i, 1).as<double>(), sph_bessel_j0(vals(i, 0).as<double>())));
    }
}

TEST(Special, BesselY0) {
    nd::array vals = bessel_y0_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(i, 1).as<double>(), bessel_y0(vals(i, 0).as<double>())));
    }
}

TEST(Special, BesselY1) {
    nd::array vals = bessel_y1_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(i, 1).as<double>(), bessel_y1(vals(i, 0).as<double>())));
    }
}

TEST(Special, BesselY) {
    nd::array vals = bessel_y_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(i, 2).as<double>(), bessel_y(vals(i, 0).as<double>(), vals(i, 1).as<double>())));
    }
}

TEST(Special, StruveH) {
    nd::array vals = struve_h_vals();
    intptr_t size = vals.get_dim_size();

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(i, 2).as<double>(), struve_h(vals(i, 0).as<double>(), vals(i, 1).as<double>())));
    }
}

/*
TEST(Special, RiccatiBesselJ0) {
    nd::array vals = riccati_bessel_j0_vals();
    intptr_t size = vals.get_shape()[1];

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(1, i).as<double>(), riccati_bessel_j0(vals(0, i).as<double>())));
    }
}

TEST(Special, RiccatiBesselJ1) {
    nd::array vals = riccati_bessel_j1_vals();
    intptr_t size = vals.get_shape()[1];

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(1, i).as<double>(), riccati_bessel_j1(vals(0, i).as<double>())));
    }
}
*/

/*
TEST(Special, Airy) {
    nd::array vals = airy_vals();
    intptr_t size = vals.get_shape()[1];

    for (int i = 0; i < size; ++i) {
        double ai, aip, bi, bip;
        airy(vals(0, i).as<double>(), ai, aip, bi, bip);

        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(1, i).as<double>(), ai));
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(2, i).as<double>(), aip));
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(3, i).as<double>(), bi));
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(4, i).as<double>(), bip));
    }
}
*/

#undef REL_ERROR_MAX
