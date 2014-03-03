//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <inc_gtest.hpp>

#include <dynd/array.hpp>
#include <dynd/array_range.hpp>

using namespace std;
using namespace dynd;

TEST(ArrayRange, Basic) {
    nd::array a;

    a = nd::range(1, 10);
    EXPECT_EQ(ndt::make_type<int32_t>(), a.get_type().get_dtype());
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(9, a.get_shape()[0]);
    for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(i+1, a(i).as<int32_t>());
    }

    a = nd::range(1., 10., 0.5);
    EXPECT_EQ(ndt::make_type<double>(), a.get_type().get_dtype());
    EXPECT_EQ(1u, a.get_shape().size());
    EXPECT_EQ(18, a.get_shape()[0]);
    for (int i = 0; i < 18; ++i) {
        EXPECT_EQ(0.5*(i+2), a(i).as<double>());
    }

    a = nd::range(0., 1., 0.1);
    EXPECT_EQ(ndt::make_type<double>(), a.get_type().get_dtype());
    EXPECT_EQ(10, a.get_shape()[0]);

    a = nd::range(0.f,1.f,0.01f);
    EXPECT_EQ(ndt::make_type<float>(), a.get_type().get_dtype());
    EXPECT_EQ(100, a.get_shape()[0]);

    a = nd::range(3 <= irange() <= 20);
    for (int i = 3; i <= 20; ++i) {
        EXPECT_EQ(i, a(i-3).as<int32_t>());
    }
}

TEST(ArrayRange, CastScalars) {
    nd::array a;

    a = nd::range(4).ucast(ndt::make_type<int32_t>());
    a = a.eval();
    EXPECT_EQ(0, a(0).as<int32_t>());
    EXPECT_EQ(1, a(1).as<int32_t>());
    EXPECT_EQ(2, a(2).as<int32_t>());
    EXPECT_EQ(3, a(3).as<int32_t>());
    a = a.ucast(ndt::make_type<double>());
    a = a.eval();
    EXPECT_EQ(0., a(0).as<double>());
    EXPECT_EQ(1., a(1).as<double>());
    EXPECT_EQ(2., a(2).as<double>());
    EXPECT_EQ(3., a(3).as<double>());
}

TEST(ArrayLinspace, Basic) {
	nd::array a;

	a = nd::linspace(0, 3, 4);
    EXPECT_EQ(ndt::type("strided * float64"), a.get_type());
    EXPECT_EQ(4u, a.get_shape()[0]);
    EXPECT_EQ(0, a(0).as<double>());
    EXPECT_EQ(1, a(1).as<double>());
    EXPECT_EQ(2, a(2).as<double>());
    EXPECT_EQ(3, a(3).as<double>());

	a = nd::linspace(0, 2, 5);
    EXPECT_EQ(ndt::type("strided * float64"), a.get_type());
    EXPECT_EQ(5u, a.get_shape()[0]);
    EXPECT_EQ(0, a(0).as<double>());
    EXPECT_EQ(0.5, a(1).as<double>());
    EXPECT_EQ(1, a(2).as<double>());
    EXPECT_EQ(1.5, a(3).as<double>());
    EXPECT_EQ(2, a(4).as<double>());

	a = nd::linspace(0.f, 1.f, 3);
    EXPECT_EQ(ndt::type("strided * float32"), a.get_type());
    EXPECT_EQ(3u, a.get_shape()[0]);
    EXPECT_EQ(0.f, a(0).as<float>());
    EXPECT_EQ(0.5f, a(1).as<float>());
    EXPECT_EQ(1.f, a(2).as<float>());

	a = nd::linspace(complex<float>(0.f, 0.f), complex<float>(0.f, 1.f), 3);
    EXPECT_EQ(ndt::type("strided * complex[float32]"), a.get_type());
    EXPECT_EQ(3u, a.get_shape()[0]);
    EXPECT_EQ(complex<float>(0.f, 0.f), a(0).as<complex<float> >());
    EXPECT_EQ(complex<float>(0.f, 0.5f), a(1).as<complex<float> >());
    EXPECT_EQ(complex<float>(0.f, 1.f), a(2).as<complex<float> >());

	a = nd::linspace(complex<double>(1.f, 0.f), complex<double>(0.f, 1.f), 3);
    EXPECT_EQ(ndt::type("strided * complex[float64]"), a.get_type());
    EXPECT_EQ(3u, a.get_shape()[0]);
    EXPECT_EQ(complex<double>(1., 0.), a(0).as<complex<double> >());
    EXPECT_EQ(complex<double>(0.5, 0.5), a(1).as<complex<double> >());
    EXPECT_EQ(complex<double>(0., 1.), a(2).as<complex<double> >());
}
