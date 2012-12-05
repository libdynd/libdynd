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
    ndobject a;

    gfunc::callable c = gfunc::make_callable(&one_parameter, "x");
    EXPECT_EQ(make_fixedstruct_dtype(make_dtype<int>(), "x"),
            c.get_parameters_dtype());
    c.debug_print(cout);
}

static double two_parameters(double a, long b) {
    return a * b;
}

TEST(GFuncCallable, TwoParameters) {
    ndobject a;

    gfunc::callable c = gfunc::make_callable(&two_parameters, "a", "b");
    EXPECT_EQ(make_fixedstruct_dtype(make_dtype<double>(), "a", make_dtype<long>(), "b"),
            c.get_parameters_dtype());
    c.debug_print(cout);
}

static complex<float> three_parameters(bool x, int a, int b) {
    if (x) {
        return complex<float>(a, b);
    } else {
        return complex<float>(b, a);
    }
}

TEST(GFuncCallable, ThreeParameters) {
    ndobject a;

    gfunc::callable c = gfunc::make_callable(&three_parameters, "x", "a", "b");
    EXPECT_EQ(make_fixedstruct_dtype(make_dtype<dynd_bool>(), "s", make_dtype<int>(), "a", make_dtype<int>(), "b"),
            c.get_parameters_dtype());
    c.debug_print(cout);
}

