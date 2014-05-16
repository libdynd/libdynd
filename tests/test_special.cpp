// Copyright (C) 2011-14 Irwin Zaid, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <inc_gtest.hpp>

#include <dynd/special.hpp>
#include <dynd/special_vals.hpp>

using namespace std;
using namespace dynd;

double rel_error(double expected, double actual) {
    if ((expected == 0.0) && (actual == 0.0)) {
        return 0.0;
    }

    return fabs(1.0 - actual / expected);
}

TEST(Special, BesselJ0) {
#define REL_ERROR_MAX 1E-6

    nd::array vals = bessel_j0_vals();
    intptr_t size = vals.get_shape()[1];

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(1, i).as<double>(), bessel_j0(vals(0, i).as<double>())));
    }

#undef REL_ERROR_MAX
}

TEST(Special, BesselJ1) {
#define REL_ERROR_MAX 1E-6

    nd::array vals = bessel_j1_vals();
    intptr_t size = vals.get_shape()[1];

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(1, i).as<double>(), bessel_j1(vals(0, i).as<double>())));
    }

#undef REL_ERROR_MAX
}

TEST(Special, BesselJInt) {
#define REL_ERROR_MAX 1E-3

    nd::array vals = bessel_j_vals<int>();
    intptr_t size = vals.get_shape()[1];

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(2, i).as<double>(), bessel_j(vals(0, i).as<int>(), vals(1, i).as<double>())));
    }

#undef REL_ERROR_MAX
}

TEST(Special, BesselJDouble) {
#define REL_ERROR_MAX 1E-6

    nd::array vals = bessel_j_vals<double>();
    intptr_t size = vals.get_shape()[1];

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(2, i).as<double>(), bessel_j(vals(0, i).as<double>(), vals(1, i).as<double>())));
    }

#undef REL_ERROR_MAX
}

TEST(Special, BesselY0) {
#define REL_ERROR_MAX 1E-6

    nd::array vals = bessel_y0_vals();
    intptr_t size = vals.get_shape()[1];

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(1, i).as<double>(), bessel_y0(vals(0, i).as<double>())));
    }

#undef REL_ERROR_MAX
}

TEST(Special, BesselY1) {
#define REL_ERROR_MAX 1E-6

    nd::array vals = bessel_y1_vals();
    intptr_t size = vals.get_shape()[1];

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(1, i).as<double>(), bessel_y1(vals(0, i).as<double>())));
    }

#undef REL_ERROR_MAX
}

TEST(Special, BesselYInt) {
#define REL_ERROR_MAX 1E-6

    nd::array vals = bessel_y_vals<int>();
    intptr_t size = vals.get_shape()[1];

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(2, i).as<double>(), bessel_y(vals(0, i).as<int>(), vals(1, i).as<double>())));
    }

#undef REL_ERROR_MAX
}

/*
TEST(Special, BesselYDouble) {
#define REL_ERROR_MAX 1E-3

    nd::array vals = bessel_y_vals<double>();
    intptr_t size = vals.get_shape()[1];

    for (int i = 0; i < size; ++i) {
        EXPECT_GE(REL_ERROR_MAX,
            rel_error(vals(2, i).as<double>(), bessel_y(vals(0, i).as<double>(), vals(1, i).as<double>())));
    }

#undef REL_ERROR_MAX
}
*/

