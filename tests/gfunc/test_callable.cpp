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
#include <dynd/gfunc/make_callable.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/dtypes/string_dtype.hpp>

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
    r = c.call_generic(a);
    EXPECT_EQ(make_dtype<int>(), r.get_dtype());
    EXPECT_EQ(36, r.as<int>());

    a.at(0).val_assign(3);
    r = c.call_generic(a);
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
    r = c.call_generic(a);
    EXPECT_EQ(make_dtype<double>(), r.get_dtype());
    EXPECT_EQ(6.75, r.as<double>());

    a.at(0).val_assign(-1.5);
    a.at(1).val_assign(2);
    r = c.call_generic(a);
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
    r = c.call_generic(a);
    EXPECT_EQ(make_dtype<complex<float> >(), r.get_dtype());
    EXPECT_EQ(complex<float>(3, 4), r.as<complex<float> >());

    a.at(0).val_assign(false);
    a.at(1).val_assign(5);
    a.at(2).val_assign(6);
    r = c.call_generic(a);
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
    r = c.call_generic(a);
    EXPECT_EQ(make_dtype<uint8_t>(), r.get_dtype());
    EXPECT_EQ(4, r.as<uint8_t>());

    a.at(0).val_assign(1);
    a.at(1).val_assign(3);
    a.at(2).val_assign(0.5);
    a.at(3).val_assign(12);
    r = c.call_generic(a);
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
    r = c.call_generic(a);
    EXPECT_EQ(make_dtype<double>(), r.get_dtype());
    EXPECT_EQ(86, r.as<double>());
}

static ndobject ndobject_return(int a, int b, int c) {
    ndobject result = make_strided_ndobject(3, make_dtype<int>());
    result.at(0).vals() = a;
    result.at(1).vals() = b;
    result.at(2).vals() = c;
    return result;
}

TEST(GFuncCallable, NDObjectReturn) {
    // Create the callable
    gfunc::callable c = gfunc::make_callable(&ndobject_return, "a", "b", "c");

    // Call it and see that it gave what we want
    ndobject a, r;
    a = ndobject(c.get_parameters_dtype());

    a.at(0).val_assign(-10);
    a.at(1).val_assign(20);
    a.at(2).val_assign(1000);
    r = c.call_generic(a);
    EXPECT_EQ(make_strided_array_dtype(make_dtype<int>()), r.get_dtype());
    EXPECT_EQ(-10, r.at(0).as<int>());
    EXPECT_EQ(20, r.at(1).as<int>());
    EXPECT_EQ(1000, r.at(2).as<int>());
}

static int ndobject_param(const ndobject& n) {
    return n.get_dtype().get_undim();
}

TEST(GFuncCallable, NDObjectParam) {
    // Create the callable
    gfunc::callable c = gfunc::make_callable(&ndobject_param, "n");

    // Call it and see that it gave what we want
    ndobject tmp;
    ndobject a, r;
    a = ndobject(c.get_parameters_dtype());

    tmp = make_strided_ndobject(2, 3, 1, make_dtype<int>());
    *(void**)a.get_ndo()->m_data_pointer = tmp.get_ndo();
    r = c.call_generic(a);
    EXPECT_EQ(make_dtype<int>(), r.get_dtype());
    EXPECT_EQ(3, r.as<int>());
}

static size_t dtype_param(const dtype& d) {
    return d.get_element_size();
}

TEST(GFuncCallable, DTypeParam) {
    // Create the callable
    gfunc::callable c = gfunc::make_callable(&dtype_param, "d");

    // Call it and see that it gave what we want
    dtype tmp;
    ndobject a, r;
    a = ndobject(c.get_parameters_dtype());

    // With an extended_dtype
    tmp = make_fixedstruct_dtype(make_dtype<complex<float> >(), "A", make_dtype<int8_t>(), "B");
    *(const void**)a.get_ndo()->m_data_pointer = tmp.extended();
    r = c.call_generic(a);
    EXPECT_EQ(make_dtype<size_t>(), r.get_dtype());
    EXPECT_EQ(12u, r.as<size_t>());

    // With a builtin dtype
    tmp = make_dtype<uint64_t>();
    *(void**)a.get_ndo()->m_data_pointer = (void *)tmp.get_type_id();
    r = c.call_generic(a);
    EXPECT_EQ(make_dtype<size_t>(), r.get_dtype());
    EXPECT_EQ(8u, r.as<size_t>());
}

static string string_return(int a, int b, int c) {
    stringstream ss;
    ss << a << ", " << b << ", " << c;
    return ss.str();
}

TEST(GFuncCallable, StringReturn) {
    // Create the callable
    gfunc::callable c = gfunc::make_callable(&string_return, "a", "b", "c");

    // Call it and see that it gave what we want
    ndobject a, r;
    a = ndobject(c.get_parameters_dtype());

    a.at(0).val_assign(-10);
    a.at(1).val_assign(20);
    a.at(2).val_assign(1000);
    r = c.call_generic(a);
    EXPECT_EQ(make_string_dtype(string_encoding_utf_8), r.get_dtype());
    EXPECT_EQ("-10, 20, 1000", r.as<string>());
}
