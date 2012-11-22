//
// Copyright (C) 2011-12, Dynamic NDObject Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <inc_gtest.hpp>

#include <dynd/ndobject.hpp>
#include <dynd/ndobject_arange.hpp>

using namespace std;
using namespace dynd;

TEST(NDObjectArange, Basic) {
    ndobject a;

    a = arange(1, 10);
    cout << a << "\n";
    a = arange(1., 10., 0.5);
    cout << a << "\n";
    cout << arange(25.f) << "\n";
    cout << arange(0.,1.,0.1) << "\n";
    cout << arange(0.f,1.f,0.1f) << "\n";
    cout << arange(0.f,1.f,0.01f) << "\n";

    cout << ndobject(2) << "\n";
    //cout << ndobject(2) * arange(20) << "\n";
    //cout << 2 * arange(20) << "\n";
    cout << arange(3 <= irange() <= 20) << "\n";

    cout << linspace(10, 20) << "\n";
    cout << linspace(0, 5.0, 10) << "\n";
}

TEST(NDObjectArange, CastScalars) {
    ndobject a;

    a = arange(4).cast_scalars(make_dtype<int32_t>());
    a = a.vals();
    EXPECT_EQ(0, a.at(0).as<int32_t>());
    EXPECT_EQ(1, a.at(1).as<int32_t>());
    EXPECT_EQ(2, a.at(2).as<int32_t>());
    EXPECT_EQ(3, a.at(3).as<int32_t>());
    a = a.cast_scalars(make_dtype<double>());
    a = a.vals();
    EXPECT_EQ(0., a.at(0).as<double>());
    EXPECT_EQ(1., a.at(1).as<double>());
    EXPECT_EQ(2., a.at(2).as<double>());
    EXPECT_EQ(3., a.at(3).as<double>());
}