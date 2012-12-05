//
// Copyright (C) 2011-12, Dynamic NDObject Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "inc_gtest.hpp"

#include <dynd/gfunc/callable.hpp>

using namespace std;
using namespace dynd;

static int one_parameter(int x) {
    return 3 * x;
}

TEST(GFuncCallable, OneParameter) {
    // Create the callable
    gfunc::callable c = gfunc::make_callable(&one_parameter, "x");
    EXPECT_EQ(make_fixedstruct_dtype(make_dtype<int>(), "x"),
            c.get_parameters_dtype());

    // Call it and see that it gave what we want
    ndobject a, r;
    a = ndobject(c.get_parameters_dtype());

    a.at(0).val_assign(12);
    r = c.call(a);
    EXPECT_EQ(make_dtype<int>(), r.get_dtype());
    EXPECT_EQ(36, r.as<int>());

    a.at(0).val_assign(3);
    r = c.call(a);
    EXPECT_EQ(make_dtype<int>(), r.get_dtype());
    EXPECT_EQ(9, r.as<int>());
}

static double two_parameters(double a, long b) {
    return a * b;
}

TEST(GFuncCallable, TwoParameters) {
    // Create the callable
    gfunc::callable c = gfunc::make_callable(&two_parameters, "a", "b");
    EXPECT_EQ(make_fixedstruct_dtype(make_dtype<double>(), "a", make_dtype<long>(), "b"),
            c.get_parameters_dtype());

    // Call it and see that it gave what we want
    ndobject a, r;
    a = ndobject(c.get_parameters_dtype());

    a.at(0).val_assign(2.25);
    a.at(1).val_assign(3);
    r = c.call(a);
    EXPECT_EQ(make_dtype<double>(), r.get_dtype());
    EXPECT_EQ(6.75, r.as<double>());

    a.at(0).val_assign(-1.5);
    a.at(1).val_assign(2);
    r = c.call(a);
    EXPECT_EQ(make_dtype<double>(), r.get_dtype());
    EXPECT_EQ(-3, r.as<double>());
}

static complex<float> three_parameters(bool x, int a, int b) {
    if (x) {
        return complex<float>(a, b);
    } else {
        return complex<float>(b, a);
    }
}

TEST(GFuncCallable, ThreeParameters) {
    // Create the callable
    gfunc::callable c = gfunc::make_callable(&three_parameters, "x", "a", "b");
    EXPECT_EQ(make_fixedstruct_dtype(make_dtype<dynd_bool>(), "s", make_dtype<int>(), "a", make_dtype<int>(), "b"),
            c.get_parameters_dtype());

    // Call it and see that it gave what we want
    ndobject a, r;
    a = ndobject(c.get_parameters_dtype());

    a.at(0).val_assign(true);
    a.at(1).val_assign(3);
    a.at(2).val_assign(4);
    r = c.call(a);
    EXPECT_EQ(make_dtype<complex<float> >(), r.get_dtype());
    EXPECT_EQ(complex<float>(3, 4), r.as<complex<float> >());

    a.at(0).val_assign(false);
    a.at(1).val_assign(5);
    a.at(2).val_assign(6);
    r = c.call(a);
    EXPECT_EQ(make_dtype<complex<float> >(), r.get_dtype());
    EXPECT_EQ(complex<float>(6, 5), r.as<complex<float> >());
}

static uint8_t four_parameters(int8_t x, int16_t y, double alpha, uint32_t z) {
    return (uint8_t)(x * (1 - alpha) + y * alpha + z);
}

TEST(GFuncCallable, FourParameters) {
    // Create the callable
    gfunc::callable c = gfunc::make_callable(&four_parameters, "x", "y", "alpha", "z");
    EXPECT_EQ(make_fixedstruct_dtype(make_dtype<int8_t>(), "x", make_dtype<int16_t>(), "y",
                    make_dtype<double>(), "alpha", make_dtype<uint32_t>(), "z"),
            c.get_parameters_dtype());

    // Call it and see that it gave what we want
    ndobject a, r;
    a = ndobject(c.get_parameters_dtype());

    a.at(0).val_assign(-1);
    a.at(1).val_assign(7);
    a.at(2).val_assign(0.25);
    a.at(3).val_assign(3);
    r = c.call(a);
    EXPECT_EQ(make_dtype<uint8_t>(), r.get_dtype());
    EXPECT_EQ(4, r.as<uint8_t>());

    a.at(0).val_assign(1);
    a.at(1).val_assign(3);
    a.at(2).val_assign(0.5);
    a.at(3).val_assign(12);
    r = c.call(a);
    EXPECT_EQ(make_dtype<uint8_t>(), r.get_dtype());
    EXPECT_EQ(14, r.as<uint8_t>());
}

static double five_parameters(float (&x)[3], uint16_t a1, uint32_t a2, uint64_t a3, double (&y)[3]) {
    return x[0] * a1 * y[0] + x[1] * a2 * y[1] + x[2] * a3 * y[2];
}

TEST(GFuncCallable, FiveParameters) {
    // Create the callable
    gfunc::callable c = gfunc::make_callable(&five_parameters, "x", "a1", "a2", "a3", "y");
    EXPECT_EQ(make_fixedstruct_dtype(make_fixedarray_dtype(make_dtype<float>(), 3), "x", make_dtype<uint16_t>(), "a1",
                    make_dtype<uint32_t>(), "a2", make_dtype<uint64_t>(), "a3",
                    make_fixedarray_dtype(make_dtype<double>(), 3), "y"),
            c.get_parameters_dtype());

    // Call it and see that it gave what we want
    ndobject a, r;
    a = ndobject(c.get_parameters_dtype());

    float f0[3] = {1, 2, 3};
    double d0[3] = {1.5, 2.5, 3.5};
    a.at(0).val_assign(f0);
    a.at(1).val_assign(2);
    a.at(2).val_assign(4);
    a.at(3).val_assign(6);
    a.at(4).val_assign(d0);
    r = c.call(a);
    EXPECT_EQ(make_dtype<double>(), r.get_dtype());
    EXPECT_EQ(86, r.as<double>());
}
